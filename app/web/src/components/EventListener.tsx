// src/components/EventListener.tsx

import { useEffect, useState } from 'react';
import { useProgram } from '../anchor/setup';

interface Event {
  timestamp: string;
  signature: string;
  user?: string;
  cameraAccount?: string;
}

export default function EventListener() {
  const program = useProgram();
  const [events, setEvents] = useState<Event[]>([]);

  useEffect(() => {
    if (!program) return;

    const listener = program.addEventListener('CameraActivated', (event: any) => {
      const newEvent: Event = {
        timestamp: new Date().toISOString(),
        signature: event.signature,
        user: event.cameraAccount?.toString(),
        cameraAccount: event.cameraAccount?.toString()
      };
      setEvents(prev => [...prev, newEvent]);
    });

    return () => {
      program.removeEventListener(listener);
    };
  }, [program]);

  if (events.length === 0) return null;

  return (
    <div className="bg-white p-4 rounded-lg border border-gray-200">
      {events.map((event, index) => (
        <div key={index} className="mb-4 last:mb-0">
          <p className="text-gray-800">Time: {event.timestamp}</p>
          <p className="text-gray-800">User: {event.user}</p>
          <p className="text-gray-800">Camera: {event.cameraAccount}</p>
        </div>
      ))}
    </div>
  );
}