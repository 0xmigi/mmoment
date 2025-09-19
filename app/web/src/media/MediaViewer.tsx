import { IPFSMedia } from "../storage/ipfs/ipfs-service";
import { PipeGalleryItem } from "../storage/pipe/pipe-gallery-service";
import { unifiedIpfsService } from "../storage/ipfs/unified-ipfs-service";
import { TimelineEvent } from "../timeline/timeline-types";
import { useDynamicContext } from "@dynamic-labs/sdk-react-core";
import { Dialog } from "@headlessui/react";
import { ArrowUpRight, Download, Trash2 } from "lucide-react";
import { useState, useRef, useEffect } from "react";

interface MediaViewerProps {
  isOpen: boolean;
  onClose: () => void;
  media: IPFSMedia | PipeGalleryItem | null;
  event?: TimelineEvent;
  onDelete?: (mediaId: string) => void;
}

export default function MediaViewer({
  isOpen,
  onClose,
  media,
  event,
  onDelete,
}: MediaViewerProps) {
  const { user, primaryWallet } = useDynamicContext();
  const [deleting, setDeleting] = useState(false);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const [startY, setStartY] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState(0);

  // Scroll to show EXIF just barely peeking when modal opens
  useEffect(() => {
    if (isOpen && scrollContainerRef.current) {
      // Small delay to ensure content is rendered
      setTimeout(() => {
        if (scrollContainerRef.current) {
          // Get the viewport height and image height
          const viewportHeight = window.innerHeight;
          const imgElement =
            scrollContainerRef.current.querySelector("img, video");
          if (imgElement) {
            const imgHeight = imgElement.getBoundingClientRect().height;
            // Scroll to show most of the image with just 40-60px of EXIF peeking at bottom
            // This gives a hint that there's more content below without obscuring the image
            const scrollPosition = Math.max(0, imgHeight - viewportHeight + 60);
            scrollContainerRef.current.scrollTop = scrollPosition;
          }
        }
      }, 50);
    }
  }, [isOpen]);

  // Handle touch/mouse events for swipe down to close
  const handleStart = (
    clientY: number,
    e: React.TouchEvent | React.MouseEvent
  ) => {
    setStartY(clientY);
    setIsDragging(true);
    setDragOffset(0);
    // Prevent default to stop bouncing
    e.preventDefault();
  };

  const handleMove = (
    clientY: number,
    e: React.TouchEvent | React.MouseEvent
  ) => {
    if (!isDragging) return;

    e.preventDefault(); // Prevent scroll bounce
    const diff = clientY - startY;

    // Only handle downward swipes
    if (diff > 0) {

      // Apply resistance to the drag (40% of actual movement)
      const offset = Math.min(diff * 0.4, 250);
      setDragOffset(offset);

      // Add opacity effect based on drag distance
      if (scrollContainerRef.current) {
        const opacity = Math.max(0.3, 1 - (offset / 250) * 0.7);
        scrollContainerRef.current.style.opacity = opacity.toString();
      }
    }
  };

  const handleEnd = () => {
    if (!isDragging) return;

    // Close if swiped down more than 80px
    if (dragOffset > 80) {
      // Animate out before closing
      if (scrollContainerRef.current) {
        scrollContainerRef.current.style.transition =
          "transform 0.2s ease-out, opacity 0.2s ease-out";
        scrollContainerRef.current.style.transform = "translateY(100%)";
        scrollContainerRef.current.style.opacity = "0";
      }

      setTimeout(() => {
        onClose();
        // Reset after close
        setDragOffset(0);
        if (scrollContainerRef.current) {
          scrollContainerRef.current.style.transform = "";
          scrollContainerRef.current.style.opacity = "";
          scrollContainerRef.current.style.transition = "";
        }
      }, 200);
    } else {
      // Snap back if not enough swipe
      setDragOffset(0);
      if (scrollContainerRef.current) {
        scrollContainerRef.current.style.opacity = "1";
      }
    }

    setIsDragging(false);
    setStartY(0);
  };

  // Reset on close
  useEffect(() => {
    if (!isOpen) {
      setDragOffset(0);
      setIsDragging(false);
      if (scrollContainerRef.current) {
        scrollContainerRef.current.style.transform = "";
        scrollContainerRef.current.style.opacity = "";
        scrollContainerRef.current.style.transition = "";
      }
    }
  }, [isOpen]);

  if (!media) return null;

  // Get social identity from user's verified credentials
  const farcasterCred = user?.verifiedCredentials?.find(
    (cred) => cred.oauthProvider === "farcaster"
  );

  const twitterCred = user?.verifiedCredentials?.find(
    (cred) => cred.oauthProvider === "twitter"
  );

  // Prioritize Farcaster over Twitter
  const primarySocialCred = farcasterCred || twitterCred;

  // Get social identity from event first, user's verified credentials as fallback
  const displayName =
    event?.user.displayName || primarySocialCred?.oauthDisplayName;
  const username = event?.user.username || primarySocialCred?.oauthUsername;
  const profileImage =
    event?.user.pfpUrl || primarySocialCred?.oauthAccountPhotos?.[0];

  // Determine social provider from event data or current user credentials
  const socialProvider = (() => {
    if (username?.includes("farcaster.xyz")) return "Farcaster";
    if (username?.includes("twitter.com")) return "X / Twitter";
    if (farcasterCred) return "Farcaster";
    if (twitterCred) return "X / Twitter";
    return null;
  })();

  // Only use wallet address as fallback if no social identity
  const displayIdentity =
    displayName ||
    (media.walletAddress ? `${media.walletAddress.slice(0, 4)}...${media.walletAddress.slice(-4)}` : 'Unknown');

  // Use event's transaction ID if available, fallback to media's transaction ID
  const transactionId = event?.transactionId || media.transactionId;

  // Function to handle media download
  const handleDownload = (url: string, filename: string) => {
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  // Function to handle media deletion
  const handleDelete = async (mediaId: string) => {
    if (!primaryWallet?.address || !media) return;

    try {
      setDeleting(true);

      let success = false;

      if (media.provider === "jetson") {
        const jetsonVideos = JSON.parse(
          localStorage.getItem("jetson-videos") || "[]"
        );
        const filteredVideos = jetsonVideos.filter(
          (video: IPFSMedia) => video.id !== mediaId
        );
        localStorage.setItem("jetson-videos", JSON.stringify(filteredVideos));
        success = true;
        console.log("Deleted Jetson video from localStorage:", mediaId);
      } else {
        console.log("🗑️ Starting IPFS media deletion for:", mediaId);
        await new Promise((resolve) => setTimeout(resolve, 500));
        success = await unifiedIpfsService.deleteMedia(
          mediaId,
          primaryWallet.address
        );
        console.log(
          "🗑️ IPFS media deletion result:",
          mediaId,
          "success:",
          success
        );

        if (success) {
          await new Promise((resolve) => setTimeout(resolve, 1000));
        }
      }

      if (success) {
        console.log("✅ MEDIA DELETION SUCCESS (MediaViewer):", mediaId);
        if (onDelete) {
          onDelete(mediaId);
        }
        onClose();
      } else {
        console.error("❌ MEDIA DELETION FAILED (MediaViewer):", mediaId);
      }
    } catch (err) {
      console.error("🗑️ MEDIA DELETION ERROR (MediaViewer):", err);
    } finally {
      setDeleting(false);
    }
  };

  return (
    <Dialog open={isOpen} onClose={onClose} className="relative z-[100]">
      {/* Full-screen container with prevent overscroll */}
      <div
        className="fixed inset-0 bg-black"
        style={{ touchAction: isDragging ? "none" : "auto" }}
      >
        <Dialog.Panel className="w-full h-full flex flex-col bg-white relative">
          {/* Fixed swipe indicator at top - only visible on mobile */}
          <div
            className="fixed top-3 left-1/2 -translate-x-1/2 z-[110] md:hidden cursor-grab active:cursor-grabbing"
            onTouchStart={(e) => handleStart(e.touches[0].clientY, e)}
            onTouchMove={(e) => handleMove(e.touches[0].clientY, e)}
            onTouchEnd={handleEnd}
            style={{ touchAction: "none" }}
          >
            <div className="w-12 h-1.5 bg-white/70 backdrop-blur-sm rounded-full shadow-lg" />
          </div>

          {/* Scrollable Content with drag transform */}
          <div
            ref={scrollContainerRef}
            className="flex-1 overflow-y-auto bg-white"
            style={{
              transform: `translateY(${dragOffset}px)`,
              transition: isDragging ? "none" : "transform 0.3s ease-out",
              touchAction: "pan-y",
            }}
          >
            {/* Media Section */}
            <div className="w-full">
              {media.type === "video" ? (
                <video
                  key={media.id}
                  src={media.url}
                  className="w-full h-auto"
                  controls
                  autoPlay
                  playsInline
                />
              ) : (
                <img
                  src={media.url}
                  alt="Media preview"
                  className="w-full h-auto"
                  onError={(e) => {
                    const img = e.target as HTMLImageElement;
                    if (media.backupUrls?.length) {
                      const currentIndex = media.backupUrls.indexOf(img.src);
                      if (currentIndex < media.backupUrls.length - 1) {
                        img.src = media.backupUrls[currentIndex + 1];
                      }
                    }
                  }}
                />
              )}
            </div>

            {/* Separator line */}
            <div className="w-full h-px bg-gray-200" />

            {/* Details Section */}
            <div className="bg-white px-4 py-4 pb-safe-b min-h-screen">
              {/* Action Buttons at top of EXIF section */}
              <div className="flex justify-end gap-2 mb-4">
                <button
                  onClick={() =>
                    handleDownload(
                      media.url,
                      `${media.id}.${media.type === "video" ? "mp4" : "jpg"}`
                    )
                  }
                  className="p-2 rounded-lg bg-gray-100 hover:bg-gray-200 transition-colors"
                  title="Download"
                >
                  <Download className="w-4 h-4 text-blue-500" />
                </button>
                <button
                  onClick={() => handleDelete(media.id)}
                  disabled={deleting}
                  className="p-2 rounded-lg bg-gray-100 hover:bg-gray-200 transition-colors disabled:opacity-50"
                  title="Delete"
                >
                  <Trash2
                    className={`w-4 h-4 text-red-500 ${
                      deleting ? "animate-spin" : ""
                    }`}
                  />
                </button>
              </div>

              {/* Core Identity Section */}
              <div className="flex items-center gap-3 pb-4 border-b border-gray-100">
                <div className="w-10 h-10 rounded-full bg-gray-100 overflow-hidden">
                  {profileImage && (
                    <img
                      src={profileImage}
                      alt={displayIdentity}
                      className="w-full h-full object-cover"
                    />
                  )}
                </div>
                <div className="flex-1">
                  <div className="text-base font-medium">{displayIdentity}</div>
                  {socialProvider && (
                    <div className="text-sm text-gray-500">
                      {username && `@${username.replace("@", "")}`}{" "}
                      <span className="text-xs text-blue-600 bg-blue-50 px-1.5 py-0.5 rounded ml-1">
                        {socialProvider}
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* Action Details */}
              <div className="py-4 border-b border-gray-100">
                <div className="text-sm font-medium text-gray-900 mb-2">
                  Action
                </div>
                <div className="bg-gray-50 px-3 py-2.5 rounded-lg">
                  <div className="text-sm text-gray-700">
                    {event?.type === "video_recorded"
                      ? "Video Recorded"
                      : event?.type === "photo_captured"
                      ? "Photo Captured"
                      : event?.type === "stream_started"
                      ? "Stream Started"
                      : event?.type === "stream_ended"
                      ? "Stream Ended"
                      : "Photo Captured"}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {new Date(media.timestamp).toLocaleString()}
                  </div>
                  {(event?.cameraId || media.cameraId) && (
                    <div className="text-xs text-gray-500 mt-1.5 flex items-center gap-1">
                      <span>Camera: </span>
                      <span className="font-mono">
                        {`${(event?.cameraId || media.cameraId || "").slice(
                          0,
                          4
                        )}...${(event?.cameraId || media.cameraId || "").slice(
                          -4
                        )}`}
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* Transaction */}
              {transactionId && (
                <div className="py-4 border-b border-gray-100">
                  <div className="text-sm font-medium text-gray-900 mb-2">
                    Transaction
                  </div>
                  <div className="bg-gray-50 px-3 py-2.5 rounded-lg flex items-center justify-between">
                    <span className="text-sm font-mono text-gray-600 truncate max-w-[60%]">
                      {`${transactionId.slice(0, 8)}...${transactionId.slice(
                        -8
                      )}`}
                    </span>
                    <a
                      href={`https://solscan.io/tx/${transactionId}?cluster=devnet`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm text-blue-600 hover:text-blue-700 transition-colors flex items-center gap-1"
                    >
                      View on Solscan <ArrowUpRight className="w-3.5 h-3.5" />
                    </a>
                  </div>
                </div>
              )}

              {/* Storage Info */}
              <div className="py-4">
                <div className="text-sm font-medium text-gray-900 mb-3">
                  Storage
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Provider</span>
                    <span className="text-gray-700">
                      {media.provider === "jetson" ? "Jetson Camera" : "IPFS"}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Media Type</span>
                    <span className="text-gray-700 capitalize">
                      {media.type}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Media ID</span>
                    <span className="text-gray-700 font-mono text-xs">{`${media.id.slice(
                      0,
                      8
                    )}...`}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </Dialog.Panel>
      </div>
    </Dialog>
  );
}
