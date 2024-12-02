// check_pins.js
import fetch from 'node-fetch';

const PINATA_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySW5mb3JtYXRpb24iOnsiaWQiOiI4MTUxMzVjMi0xYzI0LTRiMWItOTEwOC00MjU2MGFlYzJhYzMiLCJlbWFpbCI6IjB4bWlnaS5ldGhAZ21haWwuY29tIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsInBpbl9wb2xpY3kiOnsicmVnaW9ucyI6W3siZGVzaXJlZFJlcGxpY2F0aW9uQ291bnQiOjEsImlkIjoiRlJBMSJ9LHsiZGVzaXJlZFJlcGxpY2F0aW9uQ291bnQiOjEsImlkIjoiTllDMSJ9XSwidmVyc2lvbiI6MX0sIm1mYV9lbmFibGVkIjpmYWxzZSwic3RhdHVzIjoiQUNUSVZFIn0sImF1dGhlbnRpY2F0aW9uVHlwZSI6InNjb3BlZEtleSIsInNjb3BlZEtleUtleSI6Ijk4MzlkM2I2NTI4MDBjMDg1MjE5Iiwic2NvcGVkS2V5U2VjcmV0IjoiZjA2YzUyM2YyYzE3Mjg4NDA0Zjk0YTc4YjZlMTRkMzZjNWRlMDc0MmIwYTc1ZjdlNjk0YjU4MmUzOGRlZDVmZCIsImV4cCI6MTc2MTMyNTAxM30.0ucBZufTTnga5AkegFnEIbGIb-O_t_nLxTx4yBayXoQ"; // Use your actual token

async function checkPins() {
    try {
        const response = await fetch(
            'https://api.pinata.cloud/data/pinList?status=pinned',
            {
                headers: {
                    'Authorization': `Bearer ${PINATA_JWT}`
                }
            }
        );
        const data = await response.json();
        console.log('All pins:', data);
    } catch (error) {
        console.error('Check failed:', error);
    }
}

checkPins();