# Slide Copy — Desktop Camera View

## SLIDE 1 — HERO

- Label: WHAT IS MOMENT?
- Headline: A camera DePIN for proof of physical presence.
- Bullets:
  - Photo booth economics and market bigger
    — venue buys the hardware, it adds value to the space 
    — "booth" is ambient, always-on, and programmable
  - Organizers place a camera anywhere — it stays public to people present

  - Founder - Azuolas 
  - 

## SLIDE 2 — WHY DOES THIS WORK?

- Label: WHY DOES THIS WORK?
- Headline: "In-person" has value, still mostly offline
- Bullets:
  - 2023 Strava: 55M → 120M users in 6 months? post Covid — people want their physical activity tracked and shared
  - Korea: photo booths 1,000 → 3,000 locations in a year — people pay to capture physical moments (small numbers, ither there better data or just no numbers?)
  - But both are narrow — one tracks runs, the other takes photos. No general infrastructure exists for physical presence.

## SLIDE 3 — AGENTS + API

- Label: WHAT CAN BE BUILT ON THIS?
- Headline: One API call away from physical context.
- Body: Every Moment camera is an API endpoint. Agents, apps, and platforms plug into physical context the same way they plug into any other service. All organically access controlled by physical presence.
- Diagram endpoints: mmoment.xyz, Personal agents like Open Claw, Maps & event infrastructure, Human data marketplaces, Your app

## SLIDE 4 — LIVE DASHBOARD

- Label: WHAT'S HAPPENING RIGHT NOW?
- Shows: location, active session/queue, stats (check-ins, photos, active now)

## SLIDE 5 — WHY NOW (locked, Shift+I)

- Label: WHY NOW AND WHY HARDWARE?
- Headline: This just became possible.
- Bullets:
  - Demand for tracking/content is bigger than ever
  - People want to get out — post-COVID appetite for IRL, but no infra exists
  - Edge AI is finally viable — real-time CV on a $500 device
  - Crypto rails are ready — speed + low fees, gasless UX

## SLIDE 6 — THE STACK (locked, Shift+I)

- Label: HOW DOES IT WORK?
- Headline: Network of nodes, presence gated access controls.
- Bullets:
  - User physical presents literally prints out the default access controls
  - Business model copies photobooths as we know them except networked and monitizable
  - Jetson Orin Nano board, HQ 9:16(vertical) camera, case/battery 
  - Sensitive data encrypted locally with the user's own keys before it leaves the device
  - Session history and content ownership go on Solana — user-owned, permanent
  - Every camera is an endpoint — anything can plug in
