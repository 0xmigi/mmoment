import { useState, useEffect, useCallback, useRef } from "react";
import { CONFIG } from "../core/config";
import { timelineService } from "../timeline/timeline-service";
import { TimelineEvent } from "../timeline/timeline-types";
import { walrusGalleryService } from "../storage/walrus/walrus-gallery-service";
import { useDynamicContext } from "@dynamic-labs/sdk-react-core";
import { Pencil, Check, X, Calendar } from "lucide-react";
import { QueuePanel } from "./QueuePanel";

interface CameraEventData {
  eventName?: string;
  eventDescription?: string;
  eventDate?: string;
  eventStartTime?: number;
  eventEndTime?: number;
  eventType?: string;
  eventLocation?: string;
  eventStatus?: string;
}

interface DesktopEventPanelProps {
  cameraId: string;
  isOwner?: boolean;
}

function formatEventTime(startTime?: number, endTime?: number): string | null {
  if (!startTime) return null;
  const start = new Date(startTime);
  const dateStr = start.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
  const startTimeStr = start.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
  });
  if (!endTime) return `${dateStr}, ${startTimeStr}`;
  const end = new Date(endTime);
  const endTimeStr = end.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
  });
  // Same day
  if (start.toDateString() === end.toDateString()) {
    return `${dateStr}, ${startTimeStr} – ${endTimeStr}`;
  }
  const endDateStr = end.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
  return `${dateStr}, ${startTimeStr} – ${endDateStr}, ${endTimeStr}`;
}

function StatusBadge({ status }: { status?: string }) {
  if (!status || status === "upcoming") {
    return (
      <span className="inline-flex items-center gap-1.5 text-[11px] font-semibold tracking-wider uppercase text-amber-600 bg-amber-50 px-2.5 py-1 rounded-full">
        <span className="w-1.5 h-1.5 rounded-full bg-amber-500" />
        Upcoming
      </span>
    );
  }
  if (status === "live") {
    return (
      <span className="inline-flex items-center gap-1.5 text-[11px] font-semibold tracking-wider uppercase text-emerald-700 bg-emerald-50 px-2.5 py-1 rounded-full">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
        Live
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 text-[11px] font-semibold tracking-wider uppercase text-neutral-400 bg-neutral-100 px-2.5 py-1 rounded-full">
      <span className="w-1.5 h-1.5 rounded-full bg-neutral-400" />
      Ended
    </span>
  );
}

export function DesktopEventPanel({ cameraId, isOwner }: DesktopEventPanelProps) {
  const { primaryWallet } = useDynamicContext();
  const [eventData, setEventData] = useState<CameraEventData>({});
  const [editing, setEditing] = useState<"name" | "description" | null>(null);
  const [editValue, setEditValue] = useState("");
  const [stats, setStats] = useState({ checkIns: 0, photos: 0, activeNow: 0 });
  const inputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Fetch event data from backend
  useEffect(() => {
    const fetchEvent = async () => {
      try {
        const res = await fetch(`${CONFIG.BACKEND_URL}/api/camera/${cameraId}/event`);
        const data = await res.json();
        if (data.success && data.event) {
          setEventData({
            eventName: data.event.eventName,
            eventDescription: data.event.eventDescription,
            eventDate: data.event.eventDate,
            eventStartTime: data.event.eventStartTime,
            eventEndTime: data.event.eventEndTime,
            eventType: data.event.eventType,
            eventLocation: data.event.eventLocation,
            eventStatus: data.event.eventStatus,
          });
        }
      } catch (err) {
        console.error("[DesktopEventPanel] Failed to fetch event:", err);
      }
    };
    fetchEvent();
  }, [cameraId]);

  // Save event data to backend
  const saveEvent = useCallback(async (updates: Partial<CameraEventData>) => {
    const updated = { ...eventData, ...updates };
    setEventData(updated);
    try {
      await fetch(`${CONFIG.BACKEND_URL}/api/camera/${cameraId}/event`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          eventName: updated.eventName,
          eventDescription: updated.eventDescription,
          eventDate: updated.eventDate,
          eventStartTime: updated.eventStartTime,
          eventEndTime: updated.eventEndTime,
          eventType: updated.eventType,
          eventLocation: updated.eventLocation,
        }),
      });
    } catch (err) {
      console.error("[DesktopEventPanel] Failed to save event:", err);
    }
  }, [cameraId, eventData]);

  // Compute stats from timeline events + gallery
  useEffect(() => {
    const updateStats = () => {
      const { events } = timelineService.getState();
      const todayStart = new Date();
      todayStart.setHours(0, 0, 0, 0);
      const todayMs = todayStart.getTime();

      const checkIns = events.filter(
        (e: TimelineEvent) => e.type === "check_in" && e.timestamp >= todayMs
      ).length;

      const photos = events.filter(
        (e: TimelineEvent) =>
          (e.type === "photo_captured" || e.type === "video_recorded") &&
          e.timestamp >= todayMs
      ).length;

      // Active = unique users who checked in but haven't checked out
      const checkedInUsers = new Set<string>();
      const checkedOutUsers = new Set<string>();
      const sorted = [...events].sort((a, b) => a.timestamp - b.timestamp);
      for (const e of sorted) {
        if (e.type === "check_in") {
          checkedInUsers.add(e.user.address);
          checkedOutUsers.delete(e.user.address);
        } else if (e.type === "check_out" || e.type === "auto_check_out") {
          checkedOutUsers.add(e.user.address);
          checkedInUsers.delete(e.user.address);
        }
      }

      setStats({
        checkIns,
        photos,
        activeNow: checkedInUsers.size,
      });
    };

    updateStats();
    const unsub = timelineService.subscribe(() => updateStats());
    return () => unsub();
  }, []);

  // Also try to get photo count from gallery
  useEffect(() => {
    const fetchGalleryCount = async () => {
      if (!primaryWallet?.address) return;
      try {
        const files = await walrusGalleryService.getUserFiles(primaryWallet.address);
        const cameraFiles = files.filter((f) => f.cameraId === cameraId);
        if (cameraFiles.length > 0) {
          setStats((prev) => ({
            ...prev,
            photos: Math.max(prev.photos, cameraFiles.length),
          }));
        }
      } catch {
        // Gallery stats are supplementary
      }
    };
    fetchGalleryCount();
  }, [primaryWallet?.address, cameraId]);

  // Focus input when editing starts
  useEffect(() => {
    if (editing === "name" && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    } else if (editing === "description" && textareaRef.current) {
      textareaRef.current.focus();
      textareaRef.current.select();
    }
  }, [editing]);

  const startEdit = (field: "name" | "description") => {
    if (!isOwner) return;
    setEditing(field);
    setEditValue(
      field === "name"
        ? eventData.eventName || ""
        : eventData.eventDescription || ""
    );
  };

  const confirmEdit = () => {
    if (editing === "name") {
      saveEvent({ eventName: editValue });
    } else if (editing === "description") {
      saveEvent({ eventDescription: editValue });
    }
    setEditing(null);
  };

  const cancelEdit = () => {
    setEditing(null);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      confirmEdit();
    } else if (e.key === "Escape") {
      cancelEdit();
    }
  };

  const timeStr = formatEventTime(eventData.eventStartTime, eventData.eventEndTime);
  const hasEventTimes = !!eventData.eventStartTime;

  return (
    <div className="flex flex-col justify-between h-full px-10 py-8">
      {/* Middle: Event info */}
      <div className="flex-1 flex flex-col justify-center max-w-xl">
        {/* Status badge + label row */}
        <div className="flex items-center gap-3 mb-3">
          <div className="text-xs font-semibold tracking-[0.2em] uppercase text-neutral-400">
            Live at
          </div>
          {hasEventTimes && <StatusBadge status={eventData.eventStatus} />}
        </div>

        {/* Event Name */}
        {editing === "name" ? (
          <div className="flex items-center gap-2 mb-2">
            <input
              ref={inputRef}
              type="text"
              value={editValue}
              onChange={(e) => setEditValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Event name"
              className="text-4xl lg:text-5xl xl:text-6xl font-bold tracking-tight bg-transparent border-b-2 border-neutral-900 outline-none w-full text-neutral-900"
            />
            <button onClick={confirmEdit} className="p-1 hover:bg-gray-100 rounded">
              <Check className="w-5 h-5" />
            </button>
            <button onClick={cancelEdit} className="p-1 hover:bg-gray-100 rounded">
              <X className="w-5 h-5" />
            </button>
          </div>
        ) : (
          <div
            className={`group mb-2 ${isOwner ? "cursor-pointer" : ""}`}
            onClick={() => startEdit("name")}
          >
            <h1 className="text-4xl lg:text-5xl xl:text-6xl font-bold tracking-tight text-neutral-900 leading-[1.1]">
              {eventData.eventName || (
                <span className="text-gray-300">
                  {isOwner ? "Add event name" : "Live Camera"}
                </span>
              )}
              {isOwner && (
                <Pencil className="w-4 h-4 inline-block ml-3 opacity-0 group-hover:opacity-40 transition-opacity -translate-y-1" />
              )}
            </h1>
          </div>
        )}

        {/* Event Description / Date */}
        {editing === "description" ? (
          <div className="flex items-start gap-2 mt-1">
            <textarea
              ref={textareaRef}
              value={editValue}
              onChange={(e) => setEditValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Event description"
              rows={2}
              className="text-base lg:text-lg text-neutral-400 bg-transparent border-b border-neutral-400 outline-none w-full resize-none"
            />
            <button onClick={confirmEdit} className="p-1 hover:bg-gray-100 rounded mt-1">
              <Check className="w-4 h-4" />
            </button>
            <button onClick={cancelEdit} className="p-1 hover:bg-gray-100 rounded mt-1">
              <X className="w-4 h-4" />
            </button>
          </div>
        ) : (
          <div
            className={`group mt-1 ${isOwner ? "cursor-pointer" : ""}`}
            onClick={() => startEdit("description")}
          >
            <p className="text-base lg:text-lg text-neutral-400">
              {eventData.eventDescription || (
                <span className="text-gray-300">
                  {isOwner ? "Add description" : ""}
                </span>
              )}
              {isOwner && !eventData.eventDescription && (
                <Pencil className="w-3.5 h-3.5 inline-block ml-2 opacity-0 group-hover:opacity-40 transition-opacity" />
              )}
            </p>
          </div>
        )}

        {/* Event time + location */}
        {(timeStr || eventData.eventLocation) && (
          <div className="mt-3 space-y-1">
            {timeStr && (
              <p className="text-sm text-neutral-500 flex items-center gap-1.5">
                <Calendar className="w-3.5 h-3.5" />
                {timeStr}
              </p>
            )}
            {eventData.eventLocation && (
              <p className="text-sm text-neutral-400">
                {eventData.eventLocation}
              </p>
            )}
          </div>
        )}

      </div>

      {/* Queue — display only on desktop/TV */}
      <div className="mb-6">
        <QueuePanel cameraId={cameraId} isOwner={isOwner} displayOnly />
      </div>

      {/* Bottom: Stats row */}
      <div className="flex gap-12">
        <div>
          <div className="text-3xl lg:text-4xl font-bold tracking-tight text-neutral-900">
            {stats.checkIns}
          </div>
          <div className="text-sm text-neutral-400 mt-0.5">check-ins today</div>
        </div>
        <div>
          <div className="text-3xl lg:text-4xl font-bold tracking-tight text-neutral-900">
            {stats.photos}
          </div>
          <div className="text-sm text-neutral-400 mt-0.5">photos captured</div>
        </div>
        <div>
          <div className="text-3xl lg:text-4xl font-bold tracking-tight text-neutral-900">
            {stats.activeNow}
          </div>
          <div className="text-sm text-neutral-400 mt-0.5">active now</div>
        </div>
      </div>
    </div>
  );
}
