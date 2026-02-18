import { Loader2 } from 'lucide-react';
import './LoadingSpinner.css';

interface LoadingSpinnerProps {
  message?: string;
  fullScreen?: boolean;
}

export default function LoadingSpinner({
  message = 'Chargement...',
  fullScreen = false
}: LoadingSpinnerProps) {
  if (fullScreen) {
    return (
      <div className="loading-fullscreen">
        <div className="loading-content">
          <Loader2 className="w-12 h-12 animate-spin text-blue-600" />
          <p className="loading-message">{message}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="loading-inline">
      <Loader2 className="w-6 h-6 animate-spin text-blue-600" />
      <span className="loading-text">{message}</span>
    </div>
  );
}
