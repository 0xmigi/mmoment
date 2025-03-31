use anchor_lang::prelude::*;
use crate::state::*;
use crate::error::ErrorCode;

#[derive(AnchorSerialize, AnchorDeserialize)]
pub struct UpdateCameraArgs {
    pub name: Option<String>,
    pub location: Option<[i64; 2]>,
    pub model: Option<String>,
}

#[derive(Accounts)]
pub struct UpdateCamera<'info> {
    #[account(mut)]
    pub owner: Signer<'info>,
    
    #[account(
        mut,
        seeds = [b"camera", camera.metadata.name.as_bytes(), owner.key().as_ref()],
        bump = camera.bump,
        constraint = camera.owner == owner.key() @ ErrorCode::Unauthorized
    )]
    pub camera: Account<'info, CameraAccount>,
}

pub fn handler(ctx: Context<UpdateCamera>, args: UpdateCameraArgs) -> Result<()> {
    let camera = &mut ctx.accounts.camera;
    
    if let Some(name) = args.name {
        if !name.is_empty() {
            msg!("Warning: Changing the name field may cause PDA derivation issues");
            camera.metadata.name = name;
        }
    }
    
    if let Some(location) = args.location {
        camera.metadata.location = Some(location);
    }
    
    if let Some(model) = args.model {
        if !model.is_empty() {
            camera.metadata.model = model;
        }
    }
    
    camera.metadata.last_activity = Clock::get()?.unix_timestamp;
    
    msg!("Camera {} updated by owner {}", camera.metadata.name, ctx.accounts.owner.key());
    
    Ok(())
}