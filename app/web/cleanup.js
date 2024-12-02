// cleanup.js
import fetch from "node-fetch";

const PINATA_JWT =
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySW5mb3JtYXRpb24iOnsiaWQiOiI4MTUxMzVjMi0xYzI0LTRiMWItOTEwOC00MjU2MGFlYzJhYzMiLCJlbWFpbCI6IjB4bWlnaS5ldGhAZ21haWwuY29tIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsInBpbl9wb2xpY3kiOnsicmVnaW9ucyI6W3siZGVzaXJlZFJlcGxpY2F0aW9uQ291bnQiOjEsImlkIjoiRlJBMSJ9LHsiZGVzaXJlZFJlcGxpY2F0aW9uQ291bnQiOjEsImlkIjoiTllDMSJ9XSwidmVyc2lvbiI6MX0sIm1mYV9lbmFibGVkIjpmYWxzZSwic3RhdHVzIjoiQUNUSVZFIn0sImF1dGhlbnRpY2F0aW9uVHlwZSI6InNjb3BlZEtleSIsInNjb3BlZEtleUtleSI6Ijk4MzlkM2I2NTI4MDBjMDg1MjE5Iiwic2NvcGVkS2V5U2VjcmV0IjoiZjA2YzUyM2YyYzE3Mjg4NDA0Zjk0YTc4YjZlMTRkMzZjNWRlMDc0MmIwYTc1ZjdlNjk0YjU4MmUzOGRlZDVmZCIsImV4cCI6MTc2MTMyNTAxM30.0ucBZufTTnga5AkegFnEIbGIb-O_t_nLxTx4yBayXoQ"; // Use your actual token

async function aggressiveCleanup() {
  try {
    // Get all pins first
    const response = await fetch(
      "https://api.pinata.cloud/data/pinList?status=pinned&pageLimit=1000",
      {
        headers: {
          Authorization: `Bearer ${PINATA_JWT}`,
        },
      }
    );

    const data = await response.json();
    console.log("Found pins:", data.rows?.length || 0);

    if (data.rows && data.rows.length > 0) {
      for (const pin of data.rows) {
        console.log(`Deleting pin: ${pin.ipfs_pin_hash}`);
        try {
          // Try multiple times if needed
          for (let i = 0; i < 3; i++) {
            const deleteResponse = await fetch(
              `https://api.pinata.cloud/pinning/unpin/${pin.ipfs_pin_hash}`,
              {
                method: "DELETE",
                headers: {
                  Authorization: `Bearer ${PINATA_JWT}`,
                },
              }
            );

            if (deleteResponse.status === 200) {
              console.log(`Successfully deleted ${pin.ipfs_pin_hash}`);
              break;
            } else {
              console.log(`Attempt ${i + 1} failed for ${pin.ipfs_pin_hash}`);
              await new Promise((resolve) => setTimeout(resolve, 1000));
            }
          }
        } catch (error) {
          console.error(`Failed to delete ${pin.ipfs_pin_hash}:`, error);
        }
        // Wait between deletions
        await new Promise((resolve) => setTimeout(resolve, 500));
      }
    }

    // Verify all pins are gone
    const verifyResponse = await fetch(
      "https://api.pinata.cloud/data/pinList?status=pinned",
      {
        headers: {
          Authorization: `Bearer ${PINATA_JWT}`,
        },
      }
    );
    const verifyData = await verifyResponse.json();
    console.log("Remaining pins:", verifyData.rows?.length || 0);

    if (verifyData.rows?.length > 0) {
      console.log("WARNING: Some pins still remain:", verifyData.rows);
    }
  } catch (error) {
    console.error("Cleanup failed:", error);
  }
}

aggressiveCleanup();
