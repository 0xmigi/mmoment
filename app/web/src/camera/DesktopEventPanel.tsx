import { useState, useEffect, useCallback, useRef } from "react";
import { CONFIG } from "../core/config";
import { Lock, Pause, Play } from "lucide-react";
import { QueuePanel } from "./QueuePanel";
import { useQueue } from "../hooks/useQueue";

interface DesktopEventPanelProps {
  cameraId: string;
  isOwner?: boolean;
}

function formatCountdown(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function truncateAddress(address: string): string {
  return `${address.slice(0, 4)}...${address.slice(-4)}`;
}

// --- Slide Components ---


function SlideHero() {
  return (
    <div className="flex flex-col justify-center h-full px-10 py-8">
      <div className="flex-1 flex flex-col justify-center max-w-2xl">
        <div className="text-xs font-semibold tracking-[0.25em] uppercase text-[#8A8A82] mb-6">
          What Is Moment?
        </div>
        <h1 className="text-5xl font-black tracking-tighter text-[#1A1A18] leading-[1.05] mb-6">
          A camera DePIN for
          <br />
          <span className="text-[#D97706]">proof of physical presence.</span>
        </h1>
        <div className="space-y-3 max-w-lg">
          <p className="text-base text-[#5C5C56] leading-relaxed flex items-start gap-3">
            <span className="w-1.5 h-1.5 rounded-full bg-[#D97706] mt-2 flex-shrink-0" />
            Photo booth economics and market, bigger — venue buys the hardware, it adds value to the space — "booth" is ambient, always-on, and programmable
          </p>
          <p className="text-base text-[#5C5C56] leading-relaxed flex items-start gap-3">
            <span className="w-1.5 h-1.5 rounded-full bg-[#D97706] mt-2 flex-shrink-0" />
            Built for IRL organizers — events, fitness, community spaces. Place a camera anywhere, it stays public to people present
          </p>
          <p className="text-base text-[#5C5C56] leading-relaxed flex items-start gap-3">
            <span className="w-1.5 h-1.5 rounded-full bg-[#D97706] mt-2 flex-shrink-0" />
            Founder — Azuolas
          </p>
        </div>
      </div>
    </div>
  );
}

function ProductSpectrum() {
  return (
    <div className="w-full flex justify-center">
      <style>{`
        @keyframes fadeInDevice {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes questionResolve {
          0%, 40% { opacity: 1; transform: scale(1); }
          50% { opacity: 0; transform: scale(0.8); }
          60%, 100% { opacity: 0; transform: scale(0.8); }
        }
        @keyframes cameraAppear {
          0%, 50% { opacity: 0; transform: scale(0.6); }
          70% { opacity: 1; transform: scale(1.05); }
          100% { opacity: 1; transform: scale(1); }
        }
        @keyframes momentLabelAppear {
          0%, 60% { opacity: 0; }
          80%, 100% { opacity: 1; }
        }
      `}</style>
      <div className="flex items-end gap-12">
        {/* Phone — small */}
        <div className="flex flex-col items-center gap-2" style={{ animation: 'fadeInDevice 0.6s ease-out both' }}>
          <div className="w-7 h-12 rounded-md bg-[#1A1A18] flex items-center justify-center">
            <div className="w-5 h-9 rounded-sm bg-[#2A2A28]" />
          </div>
          <span className="text-[11px] text-[#8A8A82] font-medium">Smartphone</span>
          <span className="text-[10px] text-[#B0B0A8]">~$1,000</span>
        </div>

        {/* Arrow */}
        <div className="text-[#E8E8E3] text-lg mb-8">→</div>

        {/* Moment camera — replaces question mark */}
        <div className="flex flex-col items-center gap-2 relative">
          {/* Question mark that fades out */}
          <div
            className="w-14 h-14 rounded-full border-2 border-dashed border-[#D97706] flex items-center justify-center"
            style={{ animation: 'questionResolve 4s ease-in-out infinite' }}
          >
            <span className="text-xl font-bold text-[#D97706]">?</span>
          </div>
          {/* Camera that fades in */}
          <div
            className="absolute top-0 w-14 h-14 rounded-full bg-[#1A1A18] flex items-center justify-center"
            style={{ animation: 'cameraAppear 4s ease-out infinite' }}
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="white">
              <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
              <circle cx="12" cy="13" r="4" fill="#1A1A18" />
            </svg>
          </div>
          <span className="text-[11px] text-[#D97706] font-semibold mt-1" style={{ animation: 'momentLabelAppear 4s ease-out infinite' }}>Moment</span>
          <span className="text-[10px] text-[#B0B0A8]">~$500</span>
        </div>

        {/* Arrow */}
        <div className="text-[#E8E8E3] text-lg mb-8">→</div>

        {/* Photo booth — big */}
        <div className="flex flex-col items-center gap-2" style={{ animation: 'fadeInDevice 0.6s ease-out 0.2s both' }}>
          <svg width="72" height="88" viewBox="0 0 48 64" fill="none">
            {/* Booth frame */}
            <rect x="4" y="0" width="40" height="8" rx="2" fill="#8A8A82" />
            {/* Side walls */}
            <rect x="4" y="8" width="4" height="56" fill="#B0B0A8" />
            <rect x="40" y="8" width="4" height="56" fill="#B0B0A8" />
            {/* Back wall */}
            <rect x="8" y="8" width="32" height="48" fill="#E8E8E3" />
            {/* Screen/lens */}
            <circle cx="24" cy="24" r="6" fill="#B0B0A8" stroke="#8A8A82" strokeWidth="1.5" />
            <circle cx="24" cy="24" r="2.5" fill="#8A8A82" />
            {/* Seat */}
            <rect x="10" y="48" width="28" height="4" rx="1" fill="#D4D4CE" />
            {/* Curtain lines */}
            <path d="M 8 56 Q 14 52, 20 56 Q 26 60, 32 56 Q 38 52, 40 56" stroke="#B0B0A8" strokeWidth="1.5" fill="none" />
          </svg>
          <span className="text-[11px] text-[#8A8A82] font-medium">Photo booth</span>
          <span className="text-[10px] text-[#B0B0A8]">~$10,000+</span>
        </div>
      </div>
    </div>
  );
}

function SlideProblem() {
  return (
    <div className="flex flex-col justify-between h-full px-10 py-8">
      <div className="flex-1 flex flex-col justify-center max-w-2xl">
        <div className="text-xs font-semibold tracking-[0.25em] uppercase text-[#8A8A82] mb-6">
          Why Does This Work?
        </div>
        <h2 className="text-5xl font-black tracking-tighter text-[#1A1A18] leading-[1.05] mb-6">
          "In-person" has value.
          <br />
          <span className="text-[#D97706]">Not enough of it is being captured.</span>
        </h2>
        <div className="space-y-3 max-w-lg">
          <p className="text-base text-[#5C5C56] leading-relaxed flex items-start gap-3">
            <span className="w-1.5 h-1.5 rounded-full bg-[#D97706] mt-2 flex-shrink-0" />
            Strava: 55M → 120M users post-COVID — people want their physical activity tracked and shared
          </p>
          <p className="text-base text-[#5C5C56] leading-relaxed flex items-start gap-3">
            <span className="w-1.5 h-1.5 rounded-full bg-[#D97706] mt-2 flex-shrink-0" />
            Korea: photo booths 3x in one year — people pay to capture physical moments
          </p>
          <p className="text-base text-[#5C5C56] leading-relaxed flex items-start gap-3">
            <span className="w-1.5 h-1.5 rounded-full bg-[#D97706] mt-2 flex-shrink-0" />
            But both are narrow — one tracks runs, the other takes photos. No open infrastructure exists to capture it.
          </p>
          <p className="text-base text-[#5C5C56] leading-relaxed flex items-start gap-3">
            <span className="w-1.5 h-1.5 rounded-full bg-[#D97706] mt-2 flex-shrink-0" />
            Hardware sale + platform fee on transactions — photo booth economics, proven model
          </p>
        </div>
      </div>
      <ProductSpectrum />
    </div>
  );
}

function AgentFlowDiagram() {
  const endpoints = [
    "mmoment.xyz",
    "Personal agents like Open Claw",
    "Maps & event infrastructure",
    "Human data marketplaces",
    "Your app",
  ];

  const yPositions = [30, 75, 120, 165, 210];

  return (
    <div className="relative w-full flex justify-center" style={{ height: 240 }}>
      <div className="relative" style={{ width: 620, height: 240 }}>
        <style>{`
          @keyframes dashFlow {
            to { stroke-dashoffset: -24; }
          }
          .fork-flow {
            stroke-dasharray: 4 12;
            animation: dashFlow 2s linear infinite;
          }
        `}</style>

        <svg viewBox="0 0 620 240" className="absolute inset-0 w-full h-full">
          {/* Camera circle */}
          <circle cx="50" cy="120" r="44" fill="#1A1A18" />
          {/* Filled camera icon — centered: icon is 24x24, scaled to 1.6 = 38.4, offset = 50 - 19.2 = 30.8, 120 - 19.2 = 100.8 */}
          <g transform="translate(30.8, 100.8) scale(1.6)">
            <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" fill="white" />
            <circle cx="12" cy="13" r="4" fill="#1A1A18" />
          </g>

          {/* Trunk line */}
          <line x1="94" y1="120" x2="220" y2="120" stroke="#D97706" strokeWidth="1.5" strokeOpacity="0.25" />
          <line x1="94" y1="120" x2="220" y2="120" stroke="#D97706" strokeWidth="1.5" strokeOpacity="0.6" className="fork-flow" />

          {/* API label */}
          <text x="157" y="110" textAnchor="middle" fill="#D97706" fontSize="11" fontWeight="700" fontFamily="Inter, system-ui, sans-serif" letterSpacing="0.15em">API</text>

          {/* Fork branches */}
          {yPositions.map((y, i) => (
            <g key={i}>
              <path
                d={`M 220 120 C 280 120, 280 ${y}, 350 ${y}`}
                fill="none" stroke="#D97706" strokeWidth="1.5" strokeOpacity="0.2"
              />
              <path
                d={`M 220 120 C 280 120, 280 ${y}, 350 ${y}`}
                fill="none" stroke="#D97706" strokeWidth="1.5" strokeOpacity="0.6"
                className="fork-flow"
                style={{ animationDelay: `${i * 0.3}s` }}
              />
              <circle cx="350" cy={y} r="3.5" fill="#D97706" fillOpacity="0.6" />
            </g>
          ))}
        </svg>

        {/* Endpoint pills */}
        {endpoints.map((label, i) => (
          <div
            key={label}
            className="absolute flex items-center bg-white border border-[#E8E8E3] rounded-lg px-4 py-2 whitespace-nowrap"
            style={{
              left: 358,
              top: yPositions[i],
              transform: 'translateY(-50%)',
            }}
          >
            <span className="text-[13px] font-medium text-[#1A1A18]">{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function SlidePlatform() {
  return (
    <div className="flex flex-col justify-between h-full px-10 py-8">
      {/* Top: text */}
      <div className="flex-1 flex flex-col justify-center max-w-2xl">
        <div className="text-xs font-semibold tracking-[0.25em] uppercase text-[#8A8A82] mb-6">
          What Can Be Built On This?
        </div>
        <h2 className="text-5xl font-black tracking-tighter text-[#1A1A18] leading-[1.05] mb-6">
          One API call away from
          <br />
          <span className="text-[#D97706]">physical context.</span>
        </h2>
        <p className="text-base lg:text-lg text-[#5C5C56] leading-relaxed max-w-lg">
          Every Moment camera is an API endpoint. Agents, apps, and platforms plug into physical context the same way they plug into any other service. All organically access controlled by physical presence.
        </p>
      </div>

      {/* Bottom: fork diagram */}
      <AgentFlowDiagram />
    </div>
  );
}

function SlideWhyNow() {
  const forces = [
    {
      title: "Demand for tracking/content is bigger than ever",
      description: "People want their physical lives recorded, shared, and owned — from fitness to social to events.",
    },
    {
      title: "People want to get out",
      description: "Post-COVID appetite for IRL is real, but no infrastructure exists to give it a digital footprint.",
    },
    {
      title: "Edge AI is finally viable",
      description: "Real-time computer vision on a $500 device — wasn't possible two years ago.",
    },
    {
      title: "Crypto rails are ready",
      description: "Speed, low fees, gasless UX — users never need to touch a token.",
    },
  ];

  return (
    <div className="flex flex-col justify-center h-full px-10 py-8">
      <div className="flex-1 flex flex-col justify-center max-w-2xl">
        <div className="text-xs font-semibold tracking-[0.25em] uppercase text-[#8A8A82] mb-6">
          Why Now And Why Hardware?
        </div>
        <h2 className="text-5xl font-black tracking-tighter text-[#1A1A18] leading-[1.05] mb-10">
          This just
          <br />
          <span className="text-[#D97706]">became possible.</span>
        </h2>
        <div className="space-y-6">
          {forces.map((item, i) => (
            <div key={i} className="flex items-start gap-4">
              <div className="text-2xl font-bold text-[#E8E8E3] tabular-nums flex-shrink-0 w-8">
                {String(i + 1).padStart(2, "0")}
              </div>
              <div>
                <h3 className="text-lg lg:text-xl font-black tracking-tight text-[#1A1A18] mb-1">
                  {item.title}
                </h3>
                <p className="text-sm lg:text-base text-[#5C5C56] leading-relaxed">
                  {item.description}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function SlideStack() {
  const points = [
    "Physical presence is the access control — no logins, no invite links, just be there",
    "Data is encrypted locally with the user's own keys before it leaves the device",
    "Session history and content ownership go on Solana — user-owned, permanent",
    "Every camera is an endpoint — anything can plug in",
  ];

  return (
    <div className="flex flex-col justify-center h-full px-10 py-8">
      <div className="flex-1 flex flex-col justify-center max-w-2xl">
        <div className="text-xs font-semibold tracking-[0.25em] uppercase text-[#8A8A82] mb-6">
          How Does It Work?
        </div>
        <h2 className="text-5xl font-black tracking-tighter text-[#1A1A18] leading-[1.05] mb-6">
          Self-contained nodes.
          <br />
          <span className="text-[#D97706]">Trusted by design. Open by default.</span>
        </h2>
        <p className="text-xs font-medium text-[#8A8A82] tracking-wide mb-6">
          Jetson Orin Nano · HQ 9:16 camera · standalone case + battery — ~$500
        </p>
        <div className="space-y-3">
          {points.map((point, i) => (
            <p key={i} className="text-base text-[#5C5C56] leading-relaxed flex items-start gap-3">
              <span className="w-1.5 h-1.5 rounded-full bg-[#D97706] mt-2 flex-shrink-0" />
              {point}
            </p>
          ))}
        </div>
      </div>
    </div>
  );
}

// --- Main Component ---

export function DesktopEventPanel({ cameraId, isOwner }: DesktopEventPanelProps) {

  const [stats, setStats] = useState({ checkIns: 0, photos: 0, activeNow: 0 });
  const { queueState, remainingSeconds } = useQueue(cameraId);

  const activeSlot = queueState?.active || null;
  const activeTitle = activeSlot?.title || (activeSlot ? `${activeSlot.displayName || truncateAddress(activeSlot.walletAddress)}'s session` : null);
  const hasActiveSession = !!activeSlot;

  // --- Investor mode (Shift+I to toggle) ---
  const [investorMode, setInvestorMode] = useState(false);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.shiftKey && e.key === "I") {
        setInvestorMode((prev) => !prev);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // --- Carousel state ---
  const PITCH_SLIDE_COUNT = 3; // hero, how-it-works, agents
  const TOTAL_SLIDES = investorMode ? PITCH_SLIDE_COUNT + 2 + 1 : PITCH_SLIDE_COUNT + 1; // +2 investor, +1 dashboard
  const AUTO_ROTATE_MS = 45000;
  const [currentSlide, setCurrentSlide] = useState(0);
  const [autoPlay, setAutoPlay] = useState(true);
  const autoRotateRef = useRef<NodeJS.Timeout | null>(null);

  // Clamp slide index when investor mode is toggled off
  useEffect(() => {
    if (!investorMode && currentSlide >= PITCH_SLIDE_COUNT + 1) {
      setCurrentSlide(0);
    }
  }, [investorMode, currentSlide]);

  const stopAutoRotate = useCallback(() => {
    if (autoRotateRef.current) {
      clearInterval(autoRotateRef.current);
      autoRotateRef.current = null;
    }
  }, []);

  const startAutoRotate = useCallback(() => {
    stopAutoRotate();
    autoRotateRef.current = setInterval(() => {
      setCurrentSlide((prev) => (prev + 1) % TOTAL_SLIDES);
    }, AUTO_ROTATE_MS);
  }, [TOTAL_SLIDES, stopAutoRotate]);

  const toggleAutoPlay = useCallback(() => {
    setAutoPlay((prev) => {
      if (prev) {
        stopAutoRotate();
      } else {
        startAutoRotate();
      }
      return !prev;
    });
  }, [startAutoRotate, stopAutoRotate]);

  const goToSlide = useCallback((index: number) => {
    setCurrentSlide(index);
  }, []);

  useEffect(() => {
    if (autoPlay) {
      startAutoRotate();
    }
    return () => stopAutoRotate();
  }, [autoPlay, startAutoRotate, stopAutoRotate]);

  // Fetch stats from backend
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await fetch(`${CONFIG.BACKEND_URL}/api/camera/${cameraId}/event`);
        const data = await res.json();
        if (data.success && data.stats) {
          setStats(data.stats);
        }
      } catch (err) {
        console.error("[DesktopEventPanel] Failed to fetch stats:", err);
      }
    };
    fetchStats();
    const interval = setInterval(fetchStats, 30000);
    return () => clearInterval(interval);
  }, [cameraId]);

  // --- Live Dashboard Slide ---
  const slideLiveDashboard = (
    <div className="flex flex-col justify-between h-full px-10 py-8">
      {/* Top: Camera identity + active session */}
      <div className="flex-1 flex flex-col justify-center max-w-xl">
        <div className="text-xs font-semibold tracking-[0.25em] uppercase text-[#8A8A82] mb-1">
          What's Happening Right Now?
        </div>
        {queueState?.config.location && (
          <p className="text-sm text-[#8A8A82] mb-4">{queueState.config.location}</p>
        )}

        {hasActiveSession ? (
          <>
            <h1 className="text-4xl lg:text-5xl xl:text-6xl font-black tracking-tighter text-[#1A1A18] leading-[1.1] mb-2">
              {activeTitle}
            </h1>
            <div className="flex items-center gap-3 mt-1">
              {activeSlot!.profileImage ? (
                <img src={activeSlot!.profileImage} alt="" className="w-6 h-6 rounded-full object-cover" />
              ) : (
                <div className="w-6 h-6 rounded-full bg-[#E8E8E3] flex items-center justify-center">
                  <span className="text-[10px] font-medium text-[#5C5C56]">
                    {(activeSlot!.displayName?.[0] || activeSlot!.walletAddress[0]).toUpperCase()}
                  </span>
                </div>
              )}
              <span className="text-base text-[#5C5C56]">
                {activeSlot!.displayName || truncateAddress(activeSlot!.walletAddress)}
              </span>
              <span className="text-base text-[#8A8A82]">&middot;</span>
              <span className="text-base font-mono font-bold text-[#1A1A18] tabular-nums">
                {remainingSeconds != null ? formatCountdown(remainingSeconds) : '--:--'}
              </span>
              <span className="text-sm text-[#8A8A82]">remaining</span>
            </div>
          </>
        ) : (
          <h1 className="text-4xl lg:text-5xl xl:text-6xl font-black tracking-tighter text-[#E8E8E3] leading-[1.1]">
            No active session
          </h1>
        )}
      </div>

      {/* Queue */}
      <div className="mb-6">
        <QueuePanel cameraId={cameraId} isOwner={isOwner} displayOnly />
      </div>

      {/* Bottom: Stats row */}
      <div className="flex gap-12">
        <div>
          <div className="text-3xl lg:text-4xl font-bold tracking-tight text-[#1A1A18]">
            {stats.checkIns}
          </div>
          <div className="text-sm text-[#8A8A82] mt-0.5">check-ins today</div>
        </div>
        <div>
          <div className="text-3xl lg:text-4xl font-bold tracking-tight text-[#1A1A18]">
            {stats.photos}
          </div>
          <div className="text-sm text-[#8A8A82] mt-0.5">photos captured</div>
        </div>
        <div>
          <div className="text-3xl lg:text-4xl font-bold tracking-tight text-[#1A1A18]">
            {stats.activeNow}
          </div>
          <div className="text-sm text-[#8A8A82] mt-0.5">active now</div>
        </div>
      </div>
    </div>
  );

  const pitchSlides = [
    <SlideHero key="hero" />,
    <SlideProblem key="problem" />,
    <SlidePlatform key="platform" />,
  ];

  const investorSlides = [
    <SlideWhyNow key="whynow" />,
    <SlideStack key="stack" />,
  ];

  const slides = investorMode
    ? [...pitchSlides, ...investorSlides, slideLiveDashboard]
    : [...pitchSlides, slideLiveDashboard];

  return (
    <div className="relative h-full flex flex-col">
      {/* Slide content */}
      <div className="flex-1 relative overflow-hidden">
        {slides.map((slide, i) => (
          <div
            key={i}
            className="absolute inset-0 transition-opacity duration-700 ease-in-out"
            style={{
              opacity: currentSlide === i ? 1 : 0,
              pointerEvents: currentSlide === i ? "auto" : "none",
            }}
          >
            {slide}
          </div>
        ))}
      </div>

      {/* Dot navigation */}
      <div className="flex items-center justify-center gap-2 pb-6">
        {slides.map((_, i) => {
          const isInvestorSlide = investorMode && i >= PITCH_SLIDE_COUNT && i < PITCH_SLIDE_COUNT + 2;
          const isFirstInvestor = isInvestorSlide && i === PITCH_SLIDE_COUNT;
          return (
            <button
              key={i}
              onClick={() => goToSlide(i)}
              className={`group relative p-1 ${isFirstInvestor ? "ml-3" : ""}`}
            >
              {isFirstInvestor && (
                <Lock className="w-3 h-3 text-[#8A8A82] absolute -left-4 top-1/2 -translate-y-1/2" />
              )}
              <div
                className={`h-2 rounded-full transition-all duration-300 ${
                  currentSlide === i
                    ? isInvestorSlide ? "w-6 bg-[#D97706]" : "w-6 bg-[#1A1A18]"
                    : isInvestorSlide ? "w-2 bg-[#D97706]/30 hover:bg-[#D97706]/60" : "w-2 bg-[#E8E8E3] hover:bg-[#8A8A82]"
                }`}
              />
            </button>
          );
        })}
        {/* Play/pause toggle */}
        <button
          onClick={toggleAutoPlay}
          className="ml-2 p-1 rounded-full hover:bg-[#F3F3EF] transition-colors text-[#8A8A82] hover:text-[#1A1A18]"
          title={autoPlay ? "Pause auto-rotation" : "Resume auto-rotation"}
        >
          {autoPlay ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
        </button>
      </div>
    </div>
  );
}
