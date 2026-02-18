import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  MessageSquare,
  LogOut,
  PenTool,
  BookOpen,
  Lightbulb,
  Paperclip,
  Sparkles,
  ClipboardCheck,
  ArrowLeft,
  Loader2,
  RotateCcw,
  PanelRightOpen,
  PanelRightClose,
  Clock,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { useAppStore } from '../store/appStore';
import { sendChatMessage } from '../lib/api';
import 'katex/dist/katex.min.css';
import './Dashboard.css';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  isStreaming?: boolean;
}

interface ChatSession {
  id: string;
  title: string;
  mode: string;
  messages: Message[];
  timestamp: Date;
}

export default function Dashboard() {
  const navigate = useNavigate();
  const { user, selectedStream, selectedSpecialty, isOnboardingComplete, logout } = useAppStore();

  const [selectedAgent, setSelectedAgent] = useState('general');
  const [prompt, setPrompt] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [currentSessionId, setCurrentSessionId] = useState(`session-${Date.now()}`);
  const [chatStarted, setChatStarted] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [chatHistory, setChatHistory] = useState<ChatSession[]>([]);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const agents = [
    {
      id: 'general',
      label: 'توجيه ومسار',
      icon: <MessageSquare className="w-4 h-4" />,
      description: 'خطط مراجعتك رتب أولوياتك واعرف من أين تبدأ',
    },
    {
      id: 'exercise_help',
      label: 'حل تمارين',
      icon: <PenTool className="w-4 h-4" />,
      description: 'ساعدني في حل تمرين خطوة بخطوة',
    },
    {
      id: 'concept_explanation',
      label: 'شرح مفاهيم',
      icon: <Lightbulb className="w-4 h-4" />,
      description: 'شرح مبسط ومفصل للمفاهيم الصعبة',
    },
    {
      id: 'exam_prep',
      label: 'تحضير بكالوريا',
      icon: <BookOpen className="w-4 h-4" />,
      description: 'مراجعة شاملة وحل مواضيع سابقة',
    },
    {
      id: 'solution_review',
      label: 'تصحيح الحل',
      icon: <ClipboardCheck className="w-4 h-4" />,
      description: 'مراجعة وتصحيح حلك مع تقييم وفق معيار البكالوريا',
    },
  ];

  const handleLogout = () => {
    logout();
    navigate('/onboarding', { replace: true });
  };

  useEffect(() => {
    if (!isOnboardingComplete || !selectedStream) {
      navigate('/onboarding');
    }
  }, [isOnboardingComplete, selectedStream, navigate]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [prompt]);

  const sendMessage = async (text: string) => {
    if (!text.trim() || isLoading) return;

    setChatStarted(true);
    setPrompt('');

    const userMsg: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: text.trim(),
    };

    const assistantMsg: Message = {
      id: `assistant-${Date.now()}`,
      role: 'assistant',
      content: '',
      isStreaming: true,
    };

    const nextMessages = [...messages, userMsg, assistantMsg];
    setMessages(nextMessages);
    setIsLoading(true);

    try {
      const response = await sendChatMessage(
        text.trim(),
        selectedStream?.code,
        undefined,
        selectedAgent,
        currentSessionId
      );

      const finalMessages = nextMessages.map(m =>
        m.id === assistantMsg.id
          ? { ...m, content: response.response, isStreaming: false }
          : m
      );

      setMessages(finalMessages);

      // Save / update session in history
      const title = text.trim().length > 40 ? text.trim().slice(0, 40) + '...' : text.trim();
      setChatHistory(prev => {
        const existing = prev.findIndex(s => s.id === currentSessionId);
        const updated: ChatSession = {
          id: currentSessionId,
          title,
          mode: selectedAgent,
          messages: finalMessages.filter(m => !m.isStreaming),
          timestamp: new Date(),
        };
        if (existing >= 0) {
          const copy = [...prev];
          copy[existing] = updated;
          return copy;
        }
        return [updated, ...prev];
      });

    } catch {
      setMessages(prev =>
        prev.map(m =>
          m.id === assistantMsg.id
            ? { ...m, content: 'حدث خطأ. حاول مرة أخرى.', isStreaming: false }
            : m
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = () => sendMessage(prompt);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleNewChat = () => {
    setMessages([]);
    setChatStarted(false);
    setPrompt('');
    setCurrentSessionId(`session-${Date.now()}`);
  };

  const loadSession = (session: ChatSession) => {
    setMessages(session.messages);
    setSelectedAgent(session.mode);
    setCurrentSessionId(session.id);
    setChatStarted(true);
    setSidebarOpen(false);
  };

  const formatTime = (date: Date) => {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const mins = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    if (mins < 1) return 'الآن';
    if (mins < 60) return `منذ ${mins} د`;
    if (hours < 24) return `منذ ${hours} س`;
    return date.toLocaleDateString('ar-DZ');
  };

  if (!isOnboardingComplete) return null;

  const currentAgent = agents.find(a => a.id === selectedAgent);

  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <div className="header-content">
          <div className="header-right">
            <button
              className="sidebar-toggle-btn"
              onClick={() => setSidebarOpen(o => !o)}
              title={sidebarOpen ? 'أغلق السجل' : 'سجل المحادثات'}
            >
              {sidebarOpen
                ? <PanelRightClose className="w-5 h-5" />
                : <PanelRightOpen className="w-5 h-5" />}
            </button>
            <div className="user-welcome">
              <h1 className="welcome-title">أهلا {user?.fullName || 'طالب'}</h1>
              <div className="stream-badge">
                {selectedStream?.nameAr || selectedStream?.name}
                {selectedSpecialty && ` - ${selectedSpecialty.nameAr || selectedSpecialty.name}`}
              </div>
            </div>
          </div>
          <div className="header-left">
            {chatStarted && (
              <button className="new-chat-btn" onClick={handleNewChat} title="محادثة جديدة">
                <RotateCcw className="w-4 h-4" />
                <span>جديد</span>
              </button>
            )}
            <button className="logout-button" onClick={handleLogout}>
              <LogOut className="w-5 h-5" />
              <span className="logout-text">خروج</span>
            </button>
          </div>
        </div>
      </header>

      <div className="dashboard-body">
        {/* Sidebar */}
        <aside className={`history-sidebar ${sidebarOpen ? 'open' : ''}`}>
          <div className="sidebar-header">
            <Clock className="w-4 h-4" />
            <span>سجل المحادثات</span>
          </div>
          <div className="sidebar-list">
            {chatHistory.length === 0 ? (
              <p className="sidebar-empty">لا توجد محادثات بعد</p>
            ) : (
              chatHistory.map(session => (
                <button
                  key={session.id}
                  className={`sidebar-item ${session.id === currentSessionId ? 'active' : ''}`}
                  onClick={() => loadSession(session)}
                >
                  <span className="sidebar-item-title">{session.title}</span>
                  <span className="sidebar-item-time">{formatTime(session.timestamp)}</span>
                </button>
              ))
            )}
          </div>
        </aside>

        {/* Overlay for mobile */}
        {sidebarOpen && <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)} />}

        <main className="dashboard-main">

        {!chatStarted && (
          <div className="idle-wrapper">
            <div className="hero-section">
              <h2 className="hero-title">كيف يمكنني مساعدتك اليوم</h2>
              <p className="hero-subtitle">اختر النمط وابدأ</p>
            </div>

            <div className="agents-selection-area">
              {agents.map(agent => (
                <button
                  key={agent.id}
                  onClick={() => setSelectedAgent(agent.id)}
                  className={`agent-chip ${selectedAgent === agent.id ? 'active' : ''}`}
                >
                  {agent.icon}
                  <span>{agent.label}</span>
                </button>
              ))}
            </div>

            <div className="input-box">
              <div className="input-box-hint">
                <Sparkles className="w-3 h-3" />
                <span>{currentAgent?.description}</span>
              </div>
              <div className="textarea-wrapper">
                <textarea
                  ref={textareaRef}
                  className="prompt-textarea"
                  placeholder="اسألني أي شيء..."
                  value={prompt}
                  onChange={e => setPrompt(e.target.value)}
                  onKeyDown={handleKeyDown}
                  dir="rtl"
                  rows={1}
                />
                <div className="input-actions">
                  <button className="icon-btn" title="إرفاق ملف">
                    <Paperclip className="w-5 h-5" />
                  </button>
                  <button className="send-arrow-btn" onClick={handleSubmit}>
                    <ArrowLeft className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>

            <div className="quick-suggestions">
              <div className="suggestion-chips">
                <button onClick={() => sendMessage('لخص لي درس الحرب الباردة')}>لخص لي درس الحرب الباردة</button>
                <button onClick={() => sendMessage("حل المعادلة التفاضلية y' + y = 0")}>حل المعادلة y&apos; + y = 0</button>
                <button onClick={() => sendMessage('كيف أحسب كمية المادة')}>كيف أحسب كمية المادة</button>
              </div>
            </div>
          </div>
        )}

        {chatStarted && (
          <div className="chat-wrapper">
            <div className="chat-mode-bar">
              {agents.map(agent => (
                <button
                  key={agent.id}
                  onClick={() => setSelectedAgent(agent.id)}
                  className={`agent-chip small ${selectedAgent === agent.id ? 'active' : ''}`}
                >
                  {agent.icon}
                  <span>{agent.label}</span>
                </button>
              ))}
            </div>

            <div className="messages-area">
              {messages.map(msg => (
                <div key={msg.id} className={`msg-row ${msg.role}`}>
                  {msg.role === 'user' ? (
                    <div className="user-bubble">{msg.content}</div>
                  ) : (
                    <div className="assistant-bubble">
                      {msg.isStreaming ? (
                        <div className="thinking">
                          <span className="dot" />
                          <span className="dot" />
                          <span className="dot" />
                        </div>
                      ) : (
                        <ReactMarkdown
                          remarkPlugins={[remarkMath]}
                          rehypePlugins={[rehypeKatex]}
                        >
                          {msg.content}
                        </ReactMarkdown>
                      )}
                    </div>
                  )}
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>

            <div className="chat-input-area">
              <div className="input-box">
                <div className="input-box-hint">
                  <Sparkles className="w-3 h-3" />
                  <span>{currentAgent?.description}</span>
                </div>
                <div className="textarea-wrapper">
                  <textarea
                    ref={textareaRef}
                    className="prompt-textarea"
                    placeholder="اسألني أي شيء..."
                    value={prompt}
                    onChange={e => setPrompt(e.target.value)}
                    onKeyDown={handleKeyDown}
                    dir="rtl"
                    rows={1}
                    disabled={isLoading}
                  />
                  <div className="input-actions">
                    <button className="icon-btn" title="إرفاق ملف" disabled={isLoading}>
                      <Paperclip className="w-5 h-5" />
                    </button>
                    <button className="send-arrow-btn" onClick={handleSubmit} disabled={isLoading}>
                      {isLoading
                        ? <Loader2 className="w-4 h-4 animate-spin" />
                        : <ArrowLeft className="w-4 h-4" />
                      }
                    </button>
                  </div>
                </div>
              </div>
              <p className="disclaimer">
                <Sparkles className="w-3 h-3" />
                قد يخطئ المساعد أحيانا  تحقق من المعلومات المهمة.
              </p>
            </div>
          </div>
        )}

        </main>
      </div>
    </div>
  );
}
