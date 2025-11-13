import { IPFSMedia } from "../storage/ipfs/ipfs-service";
import { unifiedIpfsService } from "../storage/ipfs/unified-ipfs-service";
import { PipeGalleryItem } from "../storage/pipe/pipe-gallery-service";
import { pipeService } from "../storage/pipe/pipe-service";
import { TimelineEvent } from "../timeline/timeline-types";
import { useDynamicContext } from "@dynamic-labs/sdk-react-core";
import { Dialog } from "@headlessui/react";
import { ArrowUpRight, Download, Share2, Trash2, X } from "lucide-react";
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
  const [sharing, setSharing] = useState(false);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  // Load Pipe credentials when modal opens and wallet is available
  useEffect(() => {
    if (isOpen && primaryWallet?.address && media?.provider === "pipe") {
      pipeService.loadCredentialsForWallet(primaryWallet.address).catch((err) => {
        console.error("Failed to load Pipe credentials:", err);
      });
    }
  }, [isOpen, primaryWallet?.address, media?.provider]);

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
      } else if (media.provider === "pipe") {
        console.log("🗑️ Starting Pipe media deletion for:", mediaId);
        await new Promise((resolve) => setTimeout(resolve, 500));
        success = await pipeService.deleteMedia(
          mediaId,
          primaryWallet.address
        );
        console.log(
          "🗑️ Pipe media deletion result:",
          mediaId,
          "success:",
          success
        );

        if (success) {
          await new Promise((resolve) => setTimeout(resolve, 1000));
        }
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

  // Function to handle share link creation
  const handleShare = async () => {
    if (!primaryWallet?.address || !media || media.provider !== "pipe") return;

    try {
      setSharing(true);
      console.log("🔗 Creating share link for:", media.id);

      const result = await pipeService.createShareLink(
        media.id,
        primaryWallet.address,
        {
          title: `MMOMENT ${media.type}`,
          description: `Shared from MMOMENT at ${new Date(media.timestamp).toLocaleString()}`,
        }
      );

      if (result.success && result.shareUrl) {
        console.log("✅ Share link created:", result.shareUrl);

        // Copy to clipboard
        await navigator.clipboard.writeText(result.shareUrl);
        alert(`Share link copied to clipboard!\n\n${result.shareUrl}`);
      } else {
        console.error("❌ Failed to create share link:", result.error);
        alert(`Failed to create share link: ${result.error || "Unknown error"}`);
      }
    } catch (err) {
      console.error("🔗 SHARE LINK ERROR:", err);
      alert(`Error creating share link: ${err instanceof Error ? err.message : "Unknown error"}`);
    } finally {
      setSharing(false);
    }
  };

  return (
    <Dialog open={isOpen} onClose={onClose} className="relative z-[100]">
      {/* Full-screen container */}
      <div className="fixed inset-0 bg-black">
        <Dialog.Panel className="w-full h-full flex flex-col bg-white relative">
          {/* Fixed close button at top right */}
          <button
            onClick={onClose}
            className="fixed top-3 right-3 z-[110] p-2 rounded-lg bg-gray-100 hover:bg-gray-200 transition-colors"
            title="Close"
          >
            <X className="w-4 h-4 text-gray-700" />
          </button>

          {/* Scrollable Content */}
          <div
            ref={scrollContainerRef}
            className="flex-1 overflow-y-auto bg-white"
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
                {media.provider === "pipe" && (
                  <button
                    onClick={handleShare}
                    disabled={sharing}
                    className="p-2 rounded-lg bg-gray-100 hover:bg-gray-200 transition-colors disabled:opacity-50"
                    title="Create Public Share Link"
                  >
                    <Share2
                      className={`w-4 h-4 text-green-500 ${
                        sharing ? "animate-spin" : ""
                      }`}
                    />
                  </button>
                )}
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
