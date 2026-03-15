use anchor_lang::prelude::*;
use light_sdk::{
    account::LightAccount,
    address::v1::derive_address,
    cpi::{CpiAccounts, CpiInputs},
    instruction::{PackedAddressTreeInfo, ValidityProof},
};
use crate::state::{CameraAccount, TimelineEntry, ActivityType};
use crate::error::CameraNetworkError;

/// Event emitted when a timeline entry is created (no user info)
#[event]
pub struct TimelineEntryCreated {
    pub camera: Pubkey,
    pub entry_index: u64,
    pub activity_count: u8,
    pub timestamp: i64,
}

#[derive(Accounts)]
pub struct CreateTimelineEntry<'info> {
    /// Fee payer — backend pays gas
    #[account(mut)]
    pub payer: Signer<'info>,

    /// Device authenticator — must be camera's device key or owner
    pub device: Signer<'info>,

    /// Camera account — verified against device signer
    #[account(
        mut,
        constraint = (
            camera.device_pubkey == Some(device.key()) ||
            camera.owner == device.key()
        ) @ CameraNetworkError::Unauthorized
    )]
    pub camera: Account<'info, CameraAccount>,
}

pub fn handler<'info>(
    ctx: Context<'_, '_, '_, 'info, CreateTimelineEntry<'info>>,
    proof: ValidityProof,
    address_tree_info: PackedAddressTreeInfo,
    output_merkle_tree_index: u8,
    encrypted_payload: Vec<u8>,
    nonce: [u8; 12],
    access_grants_blob: Vec<u8>,
    activity_count: u8,
    chunk_index: u8,
    total_chunks: u8,
) -> Result<()> {
    let camera = &mut ctx.accounts.camera;
    let now = Clock::get()?.unix_timestamp;

    require!(!encrypted_payload.is_empty(), CameraNetworkError::InvalidCameraData);
    require!(total_chunks >= 1, CameraNetworkError::InvalidCameraData);
    require!(chunk_index < total_chunks, CameraNetworkError::InvalidCameraData);

    let program_id = crate::ID.into();
    let light_cpi_accounts = CpiAccounts::new(
        ctx.accounts.payer.as_ref(),
        ctx.remaining_accounts,
        crate::LIGHT_CPI_SIGNER,
    );

    // Derive deterministic address: ["timeline-entry", camera_key, entry_index]
    let entry_index = camera.activity_counter;
    let address_tree_pubkey = address_tree_info
        .get_tree_pubkey(&light_cpi_accounts)
        .map_err(|_| ErrorCode::AccountNotEnoughKeys)?;

    let (address, address_seed) = derive_address(
        &[
            b"timeline-entry",
            camera.key().as_ref(),
            &entry_index.to_le_bytes(),
        ],
        &address_tree_pubkey,
        &crate::ID,
    );

    let new_address_params = address_tree_info.into_new_address_params_packed(address_seed);

    // Create the compressed account
    let mut entry = LightAccount::<'_, TimelineEntry>::new_init(
        &program_id,
        Some(address),
        output_merkle_tree_index,
    );
    entry.camera = camera.key();
    entry.entry_index = entry_index;
    entry.timestamp = now;
    entry.activity_count = activity_count;
    entry.encrypted_payload = encrypted_payload;
    entry.nonce = nonce;
    entry.access_grants_blob = access_grants_blob;
    entry.chunk_index = chunk_index;
    entry.total_chunks = total_chunks;

    // CPI to light-system-program
    let cpi = CpiInputs::new_with_address(
        proof,
        vec![entry.to_account_info().map_err(ProgramError::from)?],
        vec![new_address_params],
    );
    cpi.invoke_light_system_program(light_cpi_accounts)
        .map_err(ProgramError::from)?;

    // Update camera stats
    camera.activity_counter = camera.activity_counter.saturating_add(1);
    camera.last_activity_at = now;
    camera.last_activity_type = ActivityType::CheckOut as u8;

    emit!(TimelineEntryCreated {
        camera: camera.key(),
        entry_index,
        activity_count,
        timestamp: now,
    });

    Ok(())
}
