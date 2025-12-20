import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react'
import './glass.css'

type ChatItem = {
  role: 'assistant' | 'user'
  content: string
  speaker?: string
  timestamp?: Date
}

type AskEvent = {
  thread_id: string
  speaker?: string
  question?: string
  current_agent?: string
}

type MessageEventData = {
  thread_id: string
  content: string
  speaker?: string
  current_agent?: string
}

type FinalEventData = {
  thread_id: string
  message: string | null
  current_agent?: string
}

type ToolEventData = {
  thread_id: string
  id: string
  name: string
  args?: Record<string, any>
  agent?: string
}

// Agent configuration with colors and icons
const AGENT_CONFIG: Record<string, { icon: string; color: string; gradient: string }> = {
  'GP': { icon: '👨‍⚕️', color: '#7bc6ff', gradient: 'linear-gradient(135deg, #7bc6ff, #5ba3d9)' },
  'Ophthalmologist': { icon: '👁️', color: '#a78bfa', gradient: 'linear-gradient(135deg, #a78bfa, #8b5cf6)' },
  'Pediatrician': { icon: '👶', color: '#f472b6', gradient: 'linear-gradient(135deg, #f472b6, #ec4899)' },
  'Orthopedist': { icon: '🦴', color: '#34d399', gradient: 'linear-gradient(135deg, #34d399, #10b981)' },
  'Dermatologist': { icon: '🧴', color: '#fbbf24', gradient: 'linear-gradient(135deg, #fbbf24, #f59e0b)' },
  'ENT': { icon: '👂', color: '#fb7185', gradient: 'linear-gradient(135deg, #fb7185, #f43f5e)' },
  'Gynecologist': { icon: '🩺', color: '#c084fc', gradient: 'linear-gradient(135deg, #c084fc, #a855f7)' },
  'Psychiatrist': { icon: '🧠', color: '#60a5fa', gradient: 'linear-gradient(135deg, #60a5fa, #3b82f6)' },
  'Internal Medicine': { icon: '💊', color: '#4ade80', gradient: 'linear-gradient(135deg, #4ade80, #22c55e)' },
  'Pathologist': { icon: '🔬', color: '#f97316', gradient: 'linear-gradient(135deg, #f97316, #ea580c)' },
  'Radiologist': { icon: '📡', color: '#22d3ee', gradient: 'linear-gradient(135deg, #22d3ee, #06b6d4)' },
  'Specialist': { icon: '⚕️', color: '#a78bfa', gradient: 'linear-gradient(135deg, #a78bfa, #8b5cf6)' },
  'AI': { icon: '🤖', color: '#94a3b8', gradient: 'linear-gradient(135deg, #94a3b8, #64748b)' },
}

const getAgentConfig = (speaker?: string) => {
  if (!speaker) return AGENT_CONFIG['AI']
  // Try exact match first
  if (AGENT_CONFIG[speaker]) return AGENT_CONFIG[speaker]
  // Try partial match
  const key = Object.keys(AGENT_CONFIG).find(k => 
    speaker.toLowerCase().includes(k.toLowerCase()) || k.toLowerCase().includes(speaker.toLowerCase())
  )
  return key ? AGENT_CONFIG[key] : AGENT_CONFIG['AI']
}

const formatTime = (date: Date) => {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

const BACKEND = import.meta.env.VITE_API_BASE || '' // use proxy when ''
const LOGIN_URL = 'http://localhost:8000/login'
const SIGNUP_URL = 'http://localhost:8000/users/'
const TOKEN_STORAGE_KEY = 'access_token'

export default function App() {
  const [threadId, setThreadId] = useState<string | null>(null)
  const [chat, setChat] = useState<ChatItem[]>([])
  const [pendingAsk, setPendingAsk] = useState<AskEvent | null>(null)
  const [currentAgent, setCurrentAgent] = useState<string>('GP')
  const [tools, setTools] = useState<ToolEventData[]>([])
  const [token, setToken] = useState<string | null>(() => localStorage.getItem(TOKEN_STORAGE_KEY))
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [name, setName] = useState('')
  const [age, setAge] = useState('')
  const [gender, setGender] = useState('')
  const [authError, setAuthError] = useState<string | null>(null)
  const [authLoading, setAuthLoading] = useState(false)
  const [authMode, setAuthMode] = useState<'login' | 'signup'>('login')
  const [isTyping, setIsTyping] = useState(false)
  const [showConfetti, setShowConfetti] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const chatRef = useRef<HTMLDivElement>(null)
  const toolListRef = useRef<HTMLDivElement>(null)

  // Cursor-driven glow & subtle tilt
  useEffect(() => {
    const el = containerRef.current ?? document.documentElement
    let raf = 0
    const handle = (e: MouseEvent) => {
      const x = e.clientX
      const y = e.clientY
      document.documentElement.style.setProperty('--mx', `${x}px`)
      document.documentElement.style.setProperty('--my', `${y}px`)
      if (containerRef.current) {
        // Parallax tilt relative to center
        const rect = containerRef.current.getBoundingClientRect()
        const cx = rect.left + rect.width / 2
        const cy = rect.top + rect.height / 2
        const dx = (x - cx) / rect.width
        const dy = (y - cy) / rect.height
        cancelAnimationFrame(raf)
        raf = requestAnimationFrame(() => {
          containerRef.current!.style.setProperty('--tiltX', `${(-dy * 4).toFixed(2)}deg`)
          containerRef.current!.style.setProperty('--tiltY', `${(dx * 6).toFixed(2)}deg`)
        })
      }
    }
    window.addEventListener('mousemove', handle)
    return () => {
      window.removeEventListener('mousemove', handle)
      cancelAnimationFrame(raf)
    }
  }, [])

  const handleLogin = useCallback(async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!email.trim() || !password.trim()) {
      setAuthError('Email and password are required.')
      return
    }
    setAuthLoading(true)
    setAuthError(null)
    try {
      const response = await fetch(LOGIN_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({ username: email.trim(), password: password.trim() }).toString()
      })
      if (!response.ok) {
        throw new Error('Invalid credentials')
      }
      const data = await response.json()
      if (!data?.access_token) {
        throw new Error('Login response missing access_token')
      }
      localStorage.setItem(TOKEN_STORAGE_KEY, data.access_token)
      setToken(data.access_token)
      setEmail('')
      setPassword('')
    } catch (err) {
      setAuthError(err instanceof Error ? err.message : 'Login failed')
    } finally {
      setAuthLoading(false)
    }
  }, [email, password])

  const handleSignup = useCallback(async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!name.trim() || !email.trim() || !password.trim() || !age.trim() || !gender.trim()) {
      setAuthError('All fields are required.')
      return
    }
    const ageNum = parseInt(age, 10)
    if (isNaN(ageNum) || ageNum < 0 || ageNum > 150) {
      setAuthError('Please enter a valid age.')
      return
    }
    setAuthLoading(true)
    setAuthError(null)
    try {
      const response = await fetch(SIGNUP_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          name: name.trim(), 
          email: email.trim(), 
          password: password.trim(),
          age: ageNum,
          gender: gender.trim()
        })
      })
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Signup failed')
      }
      // Auto-login after successful signup
      const loginResponse = await fetch(LOGIN_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({ username: email.trim(), password: password.trim() }).toString()
      })
      if (!loginResponse.ok) {
        throw new Error('Account created but login failed. Please try logging in.')
      }
      const data = await loginResponse.json()
      if (!data?.access_token) {
        throw new Error('Login response missing access_token')
      }
      localStorage.setItem(TOKEN_STORAGE_KEY, data.access_token)
      setToken(data.access_token)
      setName('')
      setEmail('')
      setPassword('')
      setAge('')
      setGender('')
      // Trigger confetti animation
      setShowConfetti(true)
      setTimeout(() => setShowConfetti(false), 3000)
    } catch (err) {
      setAuthError(err instanceof Error ? err.message : 'Signup failed')
    } finally {
      setAuthLoading(false)
    }
  }, [name, email, password, age, gender])

  const handleLogout = useCallback(() => {
    localStorage.removeItem(TOKEN_STORAGE_KEY)
    setToken(null)
    setThreadId(null)
    setChat([])
    setTools([])
    setPendingAsk(null)
    setCurrentAgent('GP')
  }, [])

  const startStream = useCallback((userText: string) => {
    const activeToken = token || localStorage.getItem(TOKEN_STORAGE_KEY)
    if (!activeToken) {
      setAuthError('Please log in to start a consultation.')
      return
    }
    const url = `${BACKEND}/api/graph/start/stream?message=${encodeURIComponent(userText)}&token=${encodeURIComponent(activeToken)}`
    const es = new EventSource(url)
    setIsTyping(true)

    es.addEventListener('thread', (e) => {
      const data = JSON.parse((e as MessageEvent).data)
      setThreadId(data.thread_id)
      setCurrentAgent('GP')
    })

    es.addEventListener('message', (e) => {
      const data = JSON.parse((e as MessageEvent).data) as MessageEventData
      if (data.current_agent) setCurrentAgent(data.current_agent)
      if (!data.content?.trim()) return // Do not render empty messages
      setIsTyping(false)
      setChat((c) => {
        const last = c[c.length - 1]
        if (last && last.role === 'assistant' && last.content === data.content) return c
        return [...c, { role: 'assistant', content: data.content, speaker: data.speaker, timestamp: new Date() }]
      })
      setIsTyping(true) // Resume typing for potential next message
    })

    es.addEventListener('tool', (e) => {
      const data = JSON.parse((e as MessageEvent).data) as ToolEventData
      setTools((prev) => prev.some(t => t.id === data.id) ? prev : [...prev, data])
    })

    es.addEventListener('ask_user', (e) => {
      const data = JSON.parse((e as MessageEvent).data) as AskEvent
      setPendingAsk(data)
      setIsTyping(false)
      if (data.current_agent) setCurrentAgent(data.current_agent)
      if (data.question?.trim()) {
        setChat((c) => {
          const last = c[c.length - 1]
          if (last && last.role === 'assistant' && last.content === data.question) return c
          return [...c, { role: 'assistant', content: data.question!, speaker: data.speaker || data.current_agent || 'AI', timestamp: new Date() }]
        })
      }
      es.close()
    })

    es.addEventListener('final', (e) => {
      const data = JSON.parse((e as MessageEvent).data) as FinalEventData
      setIsTyping(false)
      if (data.current_agent) setCurrentAgent(data.current_agent)
      if (data.message) setChat((c) => [...c, { role: 'assistant', content: data.message!, timestamp: new Date() }])
      es.close()
    })

    es.onerror = () => {
      setIsTyping(false)
      es.close()
    }
  }, [token])

  const resumeStream = useCallback((tid: string, reply: string) => {
    const activeToken = token || localStorage.getItem(TOKEN_STORAGE_KEY)
    if (!activeToken) {
      setAuthError('Session expired. Please log in again.')
      return
    }
    const url = `${BACKEND}/api/graph/resume/stream?thread_id=${encodeURIComponent(tid)}&user_reply=${encodeURIComponent(reply)}&token=${encodeURIComponent(activeToken)}`
    const es = new EventSource(url)
  setPendingAsk(null)
  setChat((c) => [...c, { role: 'user', content: reply, timestamp: new Date() }])
  setIsTyping(true)

    es.addEventListener('message', (e) => {
      const data = JSON.parse((e as MessageEvent).data) as MessageEventData
      if (data.current_agent) setCurrentAgent(data.current_agent)
      if (!data.content?.trim()) return // Do not render empty messages
      setIsTyping(false)
      setChat((c) => {
        const last = c[c.length - 1]
        if (last && last.role === 'assistant' && last.content === data.content) return c
        return [...c, { role: 'assistant', content: data.content, speaker: data.speaker, timestamp: new Date() }]
      })
      setIsTyping(true)
    })

    es.addEventListener('tool', (e) => {
      const data = JSON.parse((e as MessageEvent).data) as ToolEventData
      setTools((prev) => prev.some(t => t.id === data.id) ? prev : [...prev, data])
    })

    es.addEventListener('ask_user', (e) => {
      const data = JSON.parse((e as MessageEvent).data) as AskEvent
      setPendingAsk(data)
      setIsTyping(false)
      if (data.current_agent) setCurrentAgent(data.current_agent)
      if (data.question?.trim()) {
        setChat((c) => {
          const last = c[c.length - 1]
          if (last && last.role === 'assistant' && last.content === data.question) return c
          return [...c, { role: 'assistant', content: data.question!, speaker: data.speaker || data.current_agent || 'AI', timestamp: new Date() }]
        })
      }
      es.close()
    })

    es.addEventListener('final', (e) => {
      const data = JSON.parse((e as MessageEvent).data) as FinalEventData
      setIsTyping(false)
      if (data.current_agent) setCurrentAgent(data.current_agent)
      if (data.message) setChat((c) => [...c, { role: 'assistant', content: data.message!, timestamp: new Date() }])
      es.close()
    })

    es.onerror = () => {
      setIsTyping(false)
      es.close()
    }
  }, [token])

  const onSubmit = useCallback((e: FormEvent) => {
    e.preventDefault()
    const val = inputRef.current?.value?.trim()
    if (!val) return
    if (!(token || localStorage.getItem(TOKEN_STORAGE_KEY))) {
      setAuthError('Please log in before chatting.')
      return
    }
    if (!threadId) {
      setChat((prev: ChatItem[]) => [...prev, { role: 'user', content: val, timestamp: new Date() }])
      startStream(val)
    } else if (pendingAsk?.thread_id === threadId) {
      resumeStream(threadId, val)
    }
    if (inputRef.current) inputRef.current.value = ''
  }, [threadId, pendingAsk, startStream, resumeStream, token])

  // Auto-scroll chat & tools
  useEffect(() => { if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight }, [chat])
  useEffect(() => { if (toolListRef.current) toolListRef.current.scrollTop = toolListRef.current.scrollHeight }, [tools])

  // If not authenticated, show the auth gate
  if (!token) {
    return (
      <div className="app-root">
        <div className="liquid-bg" />
        <div className="cursor-glow" />
        {showConfetti && <div className="confetti-container">{[...Array(50)].map((_, i) => <div key={i} className="confetti" style={{ '--i': i } as React.CSSProperties} />)}</div>}
        <div ref={containerRef} className="auth-gate glass tilt fade-in">
          <div className="auth-gate-header">
            <div className="auth-logo">🏥</div>
            <h1>AI Hospital</h1>
            <p className="auth-gate-subtitle">Your AI-powered virtual medical consultation</p>
          </div>
          <div className="auth-tabs">
            <button
              className={`auth-tab ${authMode === 'login' ? 'active' : ''}`}
              onClick={() => { setAuthMode('login'); setAuthError(null) }}
            >
              Login
            </button>
            <button
              className={`auth-tab ${authMode === 'signup' ? 'active' : ''}`}
              onClick={() => { setAuthMode('signup'); setAuthError(null) }}
            >
              Sign Up
            </button>
          </div>
          {authMode === 'login' ? (
            <form className="auth-gate-form" onSubmit={handleLogin}>
              <input
                type="email"
                value={email}
                placeholder="Email"
                onChange={(e) => setEmail(e.target.value)}
                className="auth-input"
              />
              <input
                type="password"
                value={password}
                placeholder="Password"
                onChange={(e) => setPassword(e.target.value)}
                className="auth-input"
              />
              <button className="btn auth-btn" type="submit" disabled={authLoading}>
                {authLoading ? 'Signing in…' : 'Login'}
              </button>
            </form>
          ) : (
            <form className="auth-gate-form" onSubmit={handleSignup}>
              <input
                type="text"
                value={name}
                placeholder="Full Name"
                onChange={(e) => setName(e.target.value)}
                className="auth-input"
              />
              <input
                type="email"
                value={email}
                placeholder="Email"
                onChange={(e) => setEmail(e.target.value)}
                className="auth-input"
              />
              <input
                type="password"
                value={password}
                placeholder="Password"
                onChange={(e) => setPassword(e.target.value)}
                className="auth-input"
              />
              <div className="auth-row">
                <input
                  type="number"
                  value={age}
                  placeholder="Age"
                  onChange={(e) => setAge(e.target.value)}
                  className="auth-input"
                  min="0"
                  max="150"
                />
                <select
                  value={gender}
                  onChange={(e) => setGender(e.target.value)}
                  className="auth-input auth-select"
                >
                  <option value="">Gender</option>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other</option>
                </select>
              </div>
              <button className="btn auth-btn" type="submit" disabled={authLoading}>
                {authLoading ? 'Creating account…' : 'Create Account'}
              </button>
            </form>
          )}
          {authError && <p className="auth-error">{authError}</p>}
          <div className="auth-gate-footer">
            <p>Secure • Private • AI-Powered Healthcare</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="app-root">
      <div className="liquid-bg" />
      <div className="cursor-glow" />
      {showConfetti && <div className="confetti-container">{[...Array(50)].map((_, i) => <div key={i} className="confetti" style={{ '--i': i } as React.CSSProperties} />)}</div>}
      <div ref={containerRef} className="container glass tilt fade-in">
        <header className="header">
          <div className="header-brand">
            <span className="header-logo">🏥</span>
            <h1>AI Hospital</h1>
          </div>
          <div className="status-tag glass" style={{ '--agent-color': getAgentConfig(currentAgent).color } as React.CSSProperties}>
            <div className="status-icon">{getAgentConfig(currentAgent).icon}</div>
            <div className="status-body glass-inner">
              <span className="status-label">Current Agent</span>
              <span className="status-value" style={{ color: getAgentConfig(currentAgent).color }}>{currentAgent || '—'}</span>
            </div>
          </div>
          <div className="header-right">
            <div className="badge pulse">Live</div>
            <button className="btn logout-btn" type="button" onClick={handleLogout}>Logout</button>
          </div>
        </header>
        <div className="main-row">
          <main className="chat glass-inner" ref={chatRef}>
            {chat.length === 0 && !isTyping && (
              <div className="chat-welcome fade-in">
                <div className="welcome-icon">👨‍⚕️</div>
                <h2>Welcome to AI Hospital</h2>
                <p>Describe your symptoms below to start a consultation with our AI General Physician.</p>
                <div className="welcome-features">
                  <div className="feature"><span>🔒</span> Secure & Private</div>
                  <div className="feature"><span>🤖</span> AI-Powered</div>
                  <div className="feature"><span>⚡</span> Instant Response</div>
                </div>
              </div>
            )}
            {chat.map((m: ChatItem, i: number) => {
              const agentConfig = getAgentConfig(m.speaker)
              return (
                <div 
                  key={i} 
                  className={`bubble ${m.role === 'user' ? 'user' : 'assistant'} fade-slide-in`}
                  style={m.role === 'assistant' ? { '--agent-color': agentConfig.color } as React.CSSProperties : undefined}
                >
                  {m.role === 'assistant' && (
                    <div className="bubble-header">
                      <div className="agent-badge" style={{ background: agentConfig.gradient }}>
                        <span className="agent-icon">{agentConfig.icon}</span>
                        <span className="agent-name">{m.speaker || 'AI'}</span>
                      </div>
                      {m.timestamp && <span className="timestamp">{formatTime(m.timestamp)}</span>}
                    </div>
                  )}
                  {m.role === 'user' && m.timestamp && (
                    <div className="bubble-header user-header">
                      <span className="timestamp">{formatTime(m.timestamp)}</span>
                    </div>
                  )}
                  <div className="content">{m.content}</div>
                </div>
              )
            })}
            {isTyping && (
              <div className="bubble assistant typing-bubble fade-in">
                <div className="bubble-header">
                  <div className="agent-badge" style={{ background: getAgentConfig(currentAgent).gradient }}>
                    <span className="agent-icon">{getAgentConfig(currentAgent).icon}</span>
                    <span className="agent-name">{currentAgent}</span>
                  </div>
                </div>
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
            {pendingAsk && !isTyping && (
              <div className="bubble assistant ask fade-in">
                <div className="bubble-header">
                  <div className="agent-badge" style={{ background: getAgentConfig(pendingAsk.current_agent || currentAgent).gradient }}>
                    <span className="agent-icon">{getAgentConfig(pendingAsk.current_agent || currentAgent).icon}</span>
                    <span className="agent-name">{pendingAsk.speaker || 'Question'}</span>
                  </div>
                </div>
                <div className="content ask-content">
                  <span className="ask-icon">💬</span>
                  Waiting for your response…
                </div>
              </div>
            )}
          </main>
          <aside className="tool-panel glass-inner">
            <div className="tool-panel-header">
              <span>🔧</span> Tool Calls
            </div>
            <div className="tool-list" ref={toolListRef}>
              {tools.map((t, idx) => {
                const toolAgent = getAgentConfig(t.agent)
                return (
                  <div key={t.id} className={`tool-card tool-${t.name?.replace(/[^a-z0-9_]/gi,'').toLowerCase()} fade-slide-in`}>
                    {idx > 0 && <div className="tool-arrow">↓</div>}
                    <div className="tool-name">{t.name}</div>
                    <div className="tool-meta">
                      <span className="tool-agent" style={{ background: toolAgent.gradient, WebkitBackgroundClip: 'text', backgroundClip: 'text', color: 'transparent' }}>
                        {toolAgent.icon} {t.agent || 'Agent'}
                      </span>
                      {t.args && Object.keys(t.args).length > 0 && (
                        <details className="tool-args"><summary>args</summary><pre>{JSON.stringify(t.args, null, 2)}</pre></details>
                      )}
                    </div>
                  </div>
                )
              })}
              {!tools.length && <div className="tool-empty">Tool activity will appear here...</div>}
            </div>
          </aside>
        </div>
        <div className="bottom-block">
          <form onSubmit={onSubmit} className="composer glass-inner">
            <input ref={inputRef} className="input" type="text" placeholder={threadId ? 'Type your answer…' : 'Describe your symptoms…'} />
            <button className="btn send-btn" type="submit">
              <span className="btn-icon">➤</span>
              {threadId ? (pendingAsk ? 'Answer' : 'Send') : 'Start'}
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}
