// Types for the application

export interface Stream {
  id: number;
  code: string;
  name: string;
  nameAr: string;
  hasOptions: boolean;
  description?: string;
  icon?: string;
  color?: string;
}

export interface SpecialtyOption {
  code: string;
  name: string;
  nameAr: string;
  description?: string;
}

export interface Subject {
  id: number;
  code: string;
  name: string;
  nameAr: string;
  category: string;
}

export interface User {
  id?: number;
  email: string;
  fullName: string;
  streamId?: number;
  streamCode?: string;
  specialtyOption?: string;
  isOnboardingComplete: boolean;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export interface BacAverageResult {
  average: number;
  totalPoints: number;
  totalCoefficients: number;
  mention: string;
  passed: boolean;
  subjectResults: SubjectResult[];
}

export interface SubjectResult {
  subjectCode: string;
  subjectName: string;
  mark: number;
  coefficient: number;
  points: number;
}
