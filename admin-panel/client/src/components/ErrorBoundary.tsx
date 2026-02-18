import { Component, type ErrorInfo, type ReactNode } from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';
import './ErrorBoundary.css';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error, errorInfo: null };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Uncaught error:', error, errorInfo);
    this.setState({ error, errorInfo });
  }

  private handleRefresh = () => {
    window.location.reload();
  };

  private handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
  };

  public render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="error-boundary">
          <div className="error-container">
            <div className="error-icon">
              <AlertCircle className="w-16 h-16" />
            </div>

            <h1 className="error-title">Oups! Une erreur s'est produite</h1>
            <p className="error-subtitle">
              Nous sommes désolés, mais quelque chose a mal tourné.
            </p>

            {this.state.error && (
              <div className="error-details">
                <p className="error-message">{this.state.error.message}</p>
                {this.state.errorInfo && (
                  <details className="error-stack">
                    <summary>Détails techniques</summary>
                    <pre>{this.state.errorInfo.componentStack}</pre>
                  </details>
                )}
              </div>
            )}

            <div className="error-actions">
              <button
                onClick={this.handleRefresh}
                className="error-button primary"
              >
                <RefreshCw className="w-4 h-4" />
                Rafraîchir la page
              </button>

              <button
                onClick={this.handleReset}
                className="error-button secondary"
              >
                Réessayer
              </button>
            </div>

            <p className="error-footer">
              Si le problème persiste, veuillez contacter le support.
            </p>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
