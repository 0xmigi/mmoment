// cleanup.js
import fetch from 'node-fetch';

const PINATA_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySW5mb3JtYXRpb24iOnsiaWQiOiI4MTUxMzVjMi0xYzI0LTRiMWItOTEwOC00MjU2MGFlYzJhYzMiLCJlbWFpbCI6IjB4bWlnaS5ldGhAZ21haWwuY29tIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsInBpbl9wb2xpY3kiOnsicmVnaW9ucyI6W3siZGVzaXJlZFJlcGxpY2F0aW9uQ291bnQiOjEsImlkIjoiRlJBMSJ9LHsiZGVzaXJlZFJlcGxpY2F0aW9uQ291bnQiOjEsImlkIjoiTllDMSJ9XSwidmVyc2lvbiI6MX0sIm1mYV9lbmFibGVkIjpmYWxzZSwic3RhdHVzIjoiQUNUSVZFIn0sImF1dGhlbnRpY2F0aW9uVHlwZSI6InNjb3BlZEtleSIsInNjb3BlZEtleUtleSI6Ijk4MzlkM2I2NTI4MDBjMDg1MjE5Iiwic2NvcGVkS2V5U2VjcmV0IjoiZjA2YzUyM2YyYzE3Mjg4NDA0Zjk0YTc4YjZlMTRkMzZjNWRlMDc0MmIwYTc1ZjdlNjk0YjU4MmUzOGRlZDVmZCIsImV4cCI6MTc2MTMyNTAxM30.0ucBZufTTnga5AkegFnEIbGIb-O_t_nLxTx4yBayXoQ"; // Use your actual token

const HASHES_TO_DELETE = [
    "QmTc1g1JuHAgXXvEBqSdZBZBGzkAeywHGtc997fJPtjcohK",
    "QmTMqknEohuUWLFDYAisQrZBnEggkeDDRewdKknDvn2Fja",
    "Qmdr38jUToJYAFpQVbjjIn7emTsLLveSMQogsnuaGjdVpC",
    "Qmap1vcmT18j2CVEV2d1BavCSuag7uSjQ2FrvbwyYuDQJu"
];

async function deleteSpecificPins() {
    try {
        for (const hash of HASHES_TO_DELETE) {
            console.log(`Attempting to delete ${hash}...`);
            try {
                const deleteResponse = await fetch(
                    `https://api.pinata.cloud/pinning/unpin/${hash}`,
                    {
                        method: 'DELETE',
                        headers: {
                            'Authorization': `Bearer ${PINATA_JWT}`
                        }
                    }
                );
                console.log(`Delete response for ${hash}: ${deleteResponse.status}`);
            } catch (error) {
                console.error(`Failed to delete ${hash}:`, error);
            }
            // Add a small delay between deletions
            await new Promise(resolve => setTimeout(resolve, 500));
        }

        // Verify all pins are gone
        const response = await fetch(
            'https://api.pinata.cloud/data/pinList?status=pinned',
            {
                headers: {
                    'Authorization': `Bearer ${PINATA_JWT}`
                }
            }
        );
        const data = await response.json();
        console.log('\nRemaining pins:', data.rows?.length || 0);
        if (data.rows?.length > 0) {
            console.log('Remaining hashes:', data.rows.map(pin => pin.ipfs_pin_hash));
        }
    } catch (error) {
        console.error('Cleanup failed:', error);
    }
}

deleteSpecificPins();