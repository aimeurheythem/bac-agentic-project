import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Stream, User, SpecialtyOption } from '../types';

interface AppState {
  // User state
  user: User | null;
  isAuthenticated: boolean;

  // Stream selection
  selectedStream: Stream | null;
  selectedSpecialty: SpecialtyOption | null;
  availableStreams: Stream[];
  specialtyOptions: SpecialtyOption[];

  // Onboarding state
  isOnboardingComplete: boolean;
  currentOnboardingStep: number;

  // Actions
  setUser: (user: User | null) => void;
  setSelectedStream: (stream: Stream | null) => void;
  setSelectedSpecialty: (specialty: SpecialtyOption | null) => void;
  setAvailableStreams: (streams: Stream[]) => void;
  setSpecialtyOptions: (options: SpecialtyOption[]) => void;
  completeOnboarding: () => void;
  resetOnboarding: () => void;
  setCurrentOnboardingStep: (step: number) => void;
  logout: () => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      // Initial state
      user: null,
      isAuthenticated: false,
      selectedStream: null,
      selectedSpecialty: null,
      availableStreams: [],
      specialtyOptions: [],
      isOnboardingComplete: false,
      currentOnboardingStep: 0,

      // Actions
      setUser: (user) => set({
        user,
        isAuthenticated: !!user,
        isOnboardingComplete: user?.isOnboardingComplete ?? false
      }),

      setSelectedStream: (stream) => set({
        selectedStream: stream,
        selectedSpecialty: null // Reset specialty when stream changes
      }),

      setSelectedSpecialty: (specialty) => set({ selectedSpecialty: specialty }),

      setAvailableStreams: (streams) => set({ availableStreams: streams }),

      setSpecialtyOptions: (options) => set({ specialtyOptions: options }),

      completeOnboarding: () => set((state) => ({
        isOnboardingComplete: true,
        user: state.user ? { ...state.user, isOnboardingComplete: true } : null
      })),

      resetOnboarding: () => set({
        isOnboardingComplete: false,
        currentOnboardingStep: 0,
        selectedStream: null,
        selectedSpecialty: null
      }),

      setCurrentOnboardingStep: (step) => set({ currentOnboardingStep: step }),

      logout: () => set({
        user: null,
        isAuthenticated: false,
        selectedStream: null,
        selectedSpecialty: null,
        isOnboardingComplete: false,
        currentOnboardingStep: 0
      })
    }),
    {
      name: 'bac-tutor-storage',
      partialize: (state) => ({
        user: state.user,
        isAuthenticated: state.isAuthenticated,
        selectedStream: state.selectedStream,
        selectedSpecialty: state.selectedSpecialty,
        isOnboardingComplete: state.isOnboardingComplete
      })
    }
  )
);
