import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { useAppStore } from './store/appStore';
import ErrorBoundary from './components/ErrorBoundary';
import Onboarding from './pages/Onboarding';
import Dashboard from './pages/Dashboard';
import ChatInterface from './pages/ChatInterface';
import './App.css';

// Protected Route component
function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isOnboardingComplete } = useAppStore();

  if (!isOnboardingComplete) {
    return <Navigate to="/onboarding" replace />;
  }

  return <>{children}</>;
}

// Public Route component (redirects to dashboard if already onboarded)
function PublicRoute({ children }: { children: React.ReactNode }) {
  const { isOnboardingComplete } = useAppStore();

  if (isOnboardingComplete) {
    return <Navigate to="/dashboard" replace />;
  }

  return <>{children}</>;
}

function App() {
  return (
    <ErrorBoundary>
      <Router>
        <div className="app">
          <Routes>
            {/* Onboarding Route */}
            <Route
              path="/onboarding"
              element={
                <PublicRoute>
                  <Onboarding />
                </PublicRoute>
              }
            />

            {/* Dashboard Route */}
            <Route
              path="/dashboard"
              element={
                <ProtectedRoute>
                  <Dashboard />
                </ProtectedRoute>
              }
            />

            {/* Chat Route */}
            <Route
              path="/chat"
              element={
                <ProtectedRoute>
                  <ChatInterface />
                </ProtectedRoute>
              }
            />

            {/* Redirect root to onboarding or dashboard */}
            <Route
              path="/"
              element={<Navigate to="/onboarding" replace />}
            />

            {/* 404 - Redirect to onboarding */}
            <Route
              path="*"
              element={<Navigate to="/onboarding" replace />}
            />
          </Routes>
        </div>
      </Router>
    </ErrorBoundary>
  );
}

export default App;
