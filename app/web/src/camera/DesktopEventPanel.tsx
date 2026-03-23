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
          Moment
        </div>
        <h1 className="text-5xl lg:text-6xl xl:text-7xl font-black tracking-tighter text-[#1A1A18] leading-[1.05] mb-6">
          You're here.
          <br />
          <span className="text-[#D97706]">That means something.</span>
        </h1>
        <p className="text-lg lg:text-xl text-[#5C5C56] leading-relaxed max-w-lg">
          Moment cameras turn physical presence into proof. The more people who show up, the stronger the record. Think of it as a witness that never forgets.
        </p>
      </div>
    </div>
  );
}

function SlideHowItWorks() {
  const steps = [
    {
      number: "01",
      title: "Tap the tag",
      description: "There's an NFC tag near this screen. It opens check-in on your phone.",
    },
    {
      number: "02",
      title: "Check in",
      description: "Sign up in seconds. Start capturing photos and videos — sent straight to your own cloud account.",
    },
    {
      number: "03",
      title: "It remembers you",
      description: "Come back and the camera recognizes you automatically. Your content, your history, all in one place. Opt-in only.",
    },
  ];

  return (
    <div className="flex flex-col justify-center h-full px-10 py-8">
      <div className="flex-1 flex flex-col justify-center max-w-2xl">
        <div className="text-xs font-semibold tracking-[0.25em] uppercase text-[#8A8A82] mb-6">
          Try It Now
        </div>
        <div className="space-y-8">
          {steps.map((step) => (
            <div key={step.number} className="flex gap-6 items-start">
              <div className="text-3xl font-bold text-[#E8E8E3] tabular-nums flex-shrink-0 w-12">
                {step.number}
              </div>
              <div>
                <h3 className="text-xl lg:text-2xl font-black tracking-tight text-[#1A1A18] mb-1">
                  {step.title}
                </h3>
                <p className="text-base text-[#5C5C56] leading-relaxed">
                  {step.description}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
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
          Your Agent Can See Your World
        </div>
        <h2 className="text-3xl lg:text-4xl xl:text-5xl font-black tracking-tighter text-[#1A1A18] leading-[1.1] mb-6">
          One API call away from
          <br />
          <span className="text-[#D97706]">physical context.</span>
        </h2>
        <p className="text-base lg:text-lg text-[#5C5C56] leading-relaxed max-w-lg">
          Every Moment camera is an API endpoint. Agents, apps, and platforms plug into physical context the same way they plug into any other service.
        </p>
      </div>

      {/* Bottom: fork diagram */}
      <AgentFlowDiagram />
    </div>
  );
}

function SlideRaise() {
  const funds = [
    "Run a semi-permanent pilot at a fitness venue",
    "Release a developer kit like device to early audience",
    "Hire a GTM/Strategy person",
    "Legal \u2014 formation, contracts",
  ];

  return (
    <div className="flex flex-col justify-center h-full px-10 py-8">
      <div className="flex-1 flex flex-col justify-center max-w-2xl">
        <div className="text-xs font-semibold tracking-[0.25em] uppercase text-[#8A8A82] mb-6">
          Pre-Seed
        </div>
        <h2 className="text-4xl lg:text-5xl xl:text-6xl font-black tracking-tighter text-[#1A1A18] leading-[1.05] mb-3">
          $300K SAFE
        </h2>
        <p className="text-xl lg:text-2xl text-[#5C5C56] mb-10">
          $2.5M cap &middot; ~12% dilution &middot; 12–18 months runway
        </p>
        <div className="text-xs font-semibold tracking-[0.25em] uppercase text-[#8A8A82] mb-4">
          Use of Funds
        </div>
        <div className="space-y-4">
          {funds.map((item, i) => (
            <div key={i} className="flex items-start gap-4">
              <div className="w-1.5 h-1.5 rounded-full bg-[#D97706] mt-2 flex-shrink-0" />
              <p className="text-base lg:text-lg text-[#1A1A18]">{item}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function SlideTeam() {
  const highlights = [
    "3 hardware iterations shipped \u2014 Pi Zero \u2192 Pi 5 \u2192 Jetson Orin Nano",
    "Full-stack: computer vision pipeline, smart contracts, frontend, hardware",
    "Seeking: GTM co-founder, advisors with hardware or consumer experience",
  ];

  return (
    <div className="flex flex-col justify-center h-full px-10 py-8">
      <div className="flex-1 flex flex-col justify-center max-w-2xl">
        <div className="text-xs font-semibold tracking-[0.25em] uppercase text-[#8A8A82] mb-6">
          Team
        </div>
        <h2 className="text-4xl lg:text-5xl xl:text-6xl font-black tracking-tighter text-[#1A1A18] leading-[1.05] mb-3">
          Solo founder.
          <br />
          <span className="text-[#D97706]">16 months in.</span>
        </h2>
        <div className="mt-8 space-y-5">
          {highlights.map((item, i) => (
            <div key={i} className="flex items-start gap-4">
              <div className="w-1.5 h-1.5 rounded-full bg-[#D97706] mt-2 flex-shrink-0" />
              <p className="text-base lg:text-lg text-[#5C5C56] leading-relaxed">{item}</p>
            </div>
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
        <div className="text-xs font-semibold tracking-[0.2em] uppercase text-[#8A8A82] mb-1">
          Live Camera
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
    <SlideHowItWorks key="how" />,
    <SlidePlatform key="platform" />,
  ];

  const investorSlides = [
    <SlideRaise key="raise" />,
    <SlideTeam key="team" />,
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
