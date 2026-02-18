import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Send,
  ArrowLeft,
  Bot,
  User,
  Loader2,
  BookOpen,
  Sparkles,
  Trash2,
  History,
  MessageSquare
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { useAppStore } from '../store/appStore';
import { sendChatMessage } from '../lib/api';
import 'katex/dist/katex.min.css';
import './ChatInterface.css';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  contextUsed?: boolean;
  isStreaming?: boolean;
}

interface ChatSession {
  id: string;
  title: string;
  lastMessage: string;
  timestamp: Date;
}

export default function ChatInterface() {
  const navigate = useNavigate();
  const { selectedStream, selectedSpecialty, user } = useAppStore();

  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string>('');
  const [showSidebar, setShowSidebar] = useState(true);
  const [chatHistory, setChatHistory] = useState<ChatSession[]>([]);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Initialize session
  useEffect(() => {
    if (!selectedStream) {
      navigate('/onboarding');
      return;
    }

    // Generate new session ID
    setSessionId(`session-${Date.now()}`);

    // Add welcome message
    const welcomeMessage: Message = {
      id: 'welcome',
      role: 'assistant',
      content: `أهلاً! أنا مساعدك الذكي لبكالوريا الجزائر.\n\nأنا هنا لمساعدتك في شعبة **${selectedStream.nameAr || selectedStream.name}**${selectedSpecialty ? ` (تخصص: ${selectedSpecialty.nameAr || selectedSpecialty.name})` : ''}.\n\nيمكنك أن تسألني عن:\n- تمارين بعينها\n- مفاهيم نظرية\n- مواضيع بكالوريا السنوات الماضية\n- طرق الحل والمنهجية\n\nكيف يمكنني مساعدتك اليوم؟`,
      timestamp: new Date(),
      contextUsed: false
    };

    setMessages([welcomeMessage]);
  }, [selectedStream, selectedSpecialty, navigate]);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [inputValue]);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: inputValue.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    // Add placeholder for assistant response
    const assistantPlaceholder: Message = {
      id: `assistant-${Date.now()}`,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isStreaming: true
    };

    setMessages(prev => [...prev, assistantPlaceholder]);

    try {
      // Call API
      const response = await sendChatMessage(
        userMessage.content,
        selectedStream?.code,
        undefined, // subject code - could be auto-detected
        'general',
        sessionId
      );

      // Update assistant message with response
      setMessages(prev =>
        prev.map(msg =>
          msg.id === assistantPlaceholder.id
            ? {
                ...msg,
                content: response.response,
                contextUsed: response.context_used,
                isStreaming: false
              }
            : msg
        )
      );

      // Update chat history
      updateChatHistory(userMessage.content, response.response);

    } catch (error) {
      console.error('Chat error:', error);

      // Show error message
      setMessages(prev =>
        prev.map(msg =>
          msg.id === assistantPlaceholder.id
            ? {
                ...msg,
                content: 'Désolé, une erreur s\'est produite. Veuillez réessayer.',
                isStreaming: false
              }
            : msg
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const updateChatHistory = (userMsg: string, assistantMsg: string) => {
    setChatHistory(prev => {
      const existingIndex = prev.findIndex(h => h.id === sessionId);
      const title = userMsg.length > 30 ? userMsg.substring(0, 30) + '...' : userMsg;

      if (existingIndex >= 0) {
        const updated = [...prev];
        updated[existingIndex] = {
          ...updated[existingIndex],
          title,
          lastMessage: assistantMsg.substring(0, 50) + '...',
          timestamp: new Date()
        };
        return updated;
      }

      return [{
        id: sessionId,
        title,
        lastMessage: assistantMsg.substring(0, 50) + '...',
        timestamp: new Date()
      }, ...prev];
    });
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const clearChat = () => {
    setMessages([{
      id: 'welcome-new',
      role: 'assistant',
      content: 'تم حذف المحادثة. كيف يمكنني مساعدتك؟',
      timestamp: new Date(),
      contextUsed: false
    }]);
    setSessionId(`session-${Date.now()}`);
  };

  const renderMessageContent = (content: string) => {
    return (
      <ReactMarkdown
        remarkPlugins={[remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          code({ node, inline, className, children, ...props }: any) {
            return (
              <code className={className} {...props}>
                {children}
              </code>
            );
          }
        }}
      >
        {content}
      </ReactMarkdown>
    );
  };

  return (
    <div className="chat-container">
      {/* Sidebar */}
      {showSidebar && (
        <aside className="chat-sidebar">
          <div className="sidebar-header">
            <button
              onClick={() => navigate('/dashboard')}
              className="sidebar-back-button"
            >
              <ArrowLeft className="w-4 h-4" />
              رجوع
            </button>
            <button
              onClick={() => setShowSidebar(false)}
              className="sidebar-close-button"
            >
              ✕
            </button>
          </div>

          <div className="sidebar-new-chat">
            <button
              onClick={clearChat}
              className="new-chat-button"
            >
              <MessageSquare className="w-4 h-4" />
              محادثة جديدة
            </button>
          </div>

          <div className="sidebar-history">
            <h3 className="history-title">
              <History className="w-4 h-4" />
              السجل السابق
            </h3>
            <div className="history-list">
              {chatHistory.length === 0 ? (
                <p className="history-empty">لا توجد محادثات</p>
              ) : (
                chatHistory.map((chat) => (
                  <button
                    key={chat.id}
                    className={`history-item ${chat.id === sessionId ? 'active' : ''}`}
                    onClick={() => setSessionId(chat.id)}
                  >
                    <span className="history-item-title">{chat.title}</span>
                    <span className="history-item-date">
                      {chat.timestamp.toLocaleDateString('ar-DZ')}
                    </span>
                  </button>
                ))
              )}
            </div>
          </div>

          <div className="sidebar-footer">
            <div className="user-info">
              <div className="user-avatar">
                {user?.fullName?.charAt(0) || 'م'}
              </div>
              <span className="user-name">{user?.fullName || 'مستخدم'}</span>
            </div>
          </div>
        </aside>
      )}

      {/* Main Chat Area */}
      <main className={`chat-main ${showSidebar ? '' : 'full-width'}`}>
        {/* Header */}
        <header className="chat-header">
          {!showSidebar && (
            <button
              onClick={() => setShowSidebar(true)}
              className="sidebar-toggle"
            >
              <History className="w-5 h-5" />
            </button>
          )}

          <div className="header-center">
            <Bot className="w-6 h-6 text-blue-600" />
            <div>
              <h1 className="header-title">المساعد الذكي</h1>
              <span className="header-stream">{selectedStream?.nameAr || selectedStream?.name}</span>
            </div>
          </div>

          <div className="header-actions">
            <button
              onClick={clearChat}
              className="header-action-button"
              title="حذف المحادثة"
            >
              <Trash2 className="w-5 h-5" />
            </button>
          </div>
        </header>

        {/* Messages Area */}
        <div className="messages-container">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`message ${message.role} ${message.isStreaming ? 'streaming' : ''}`}
            >
              <div className="message-avatar">
                {message.role === 'assistant' ? (
                  <Bot className="w-5 h-5" />
                ) : (
                  <User className="w-5 h-5" />
                )}
              </div>

              <div className="message-content">
                <div className="message-header">
                  <span className="message-author">
                    {message.role === 'assistant' ? 'المساعد الذكي' : 'أنت'}
                  </span>
                  <span className="message-time">
                    {message.timestamp.toLocaleTimeString('ar-DZ', {
                      hour: '2-digit',
                      minute: '2-digit'
                    })}
                  </span>
                  {message.contextUsed && (
                    <span className="context-badge" title="إجابة مبنية على المنهج الرسمي">
                      <BookOpen className="w-3 h-3" />
                      منهج
                    </span>
                  )}
                </div>

                <div className="message-body">
                  {message.isStreaming ? (
                    <div className="streaming-indicator">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>المساعد يفكر...</span>
                    </div>
                  ) : (
                    renderMessageContent(message.content)
                  )}
                </div>
              </div>
            </div>
          ))}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="input-container">
          <div className="input-wrapper">
            <textarea
              ref={textareaRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="اكتب سؤالك هنا... (Shift+Enter لسطر جديد)"
              className="chat-input"
              rows={1}
              disabled={isLoading}
            />

            <div className="input-actions">
              {inputValue.length > 0 && (
                <span className="input-hint">
                  {inputValue.length} حرف
                </span>
              )}

              <button
                onClick={handleSendMessage}
                disabled={!inputValue.trim() || isLoading}
                className="send-button"
              >
                {isLoading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <Send className="w-5 h-5" />
                )}
              </button>
            </div>
          </div>

          <p className="input-disclaimer">
            <Sparkles className="w-3 h-3" />
            قد يخطئ المساعد أحياناً. تحقّق من المعلومات المهمة.
          </p>
        </div>
      </main>
    </div>
  );
}
