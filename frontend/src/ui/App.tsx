import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react'
import './glass.css'

type ChatItem = {
  role: 'assistant' | 'user'
  content: string
  speaker?: string
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

const BACKEND = import.meta.env.VITE_API_BASE || '' // use proxy when ''
const LOGIN_URL = 'http://localhost:8000/login'

export default function App() {
  const [threadId, setThreadId] = useState<string | null>(null)
  const [chat, setChat] = useState<ChatItem[]>([])
  const [pendingAsk, setPendingAsk] = useState<AskEvent | null>(null)
  const [currentAgent, setCurrentAgent] = useState<string>('GP')
  const [tools, setTools] = useState<ToolEventData[]>([])
  const [token, setToken] = useState<string | null>(() => localStorage.getItem('token'))
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [authError, setAuthError] = useState<string | null>(null)
  const [authLoading, setAuthLoading] = useState(false)
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
      localStorage.setItem('token', data.access_token)
      setToken(data.access_token)
      setEmail('')
      setPassword('')
    } catch (err) {
      setAuthError(err instanceof Error ? err.message : 'Login failed')
    } finally {
      setAuthLoading(false)
    }
  }, [email, password])

  const handleLogout = useCallback(() => {
    localStorage.removeItem('token')
    setToken(null)
  }, [])

  const startStream = useCallback((userText: string) => {
    const activeToken = token || localStorage.getItem('token')
    if (!activeToken) {
      setAuthError('Please log in to start a consultation.')
      return
    }
    const url = `${BACKEND}/api/graph/start/stream?message=${encodeURIComponent(userText)}&token=${encodeURIComponent(activeToken)}`
    const es = new EventSource(url)

    es.addEventListener('thread', (e) => {
      const data = JSON.parse((e as MessageEvent).data)
      setThreadId(data.thread_id)
      setCurrentAgent('GP')
    })

    es.addEventListener('message', (e) => {
      const data = JSON.parse((e as MessageEvent).data) as MessageEventData
      if (data.current_agent) setCurrentAgent(data.current_agent)
      if (!data.content?.trim()) return // Do not render empty messages
      setChat((c) => {
        const last = c[c.length - 1]
        if (last && last.role === 'assistant' && last.content === data.content) return c
        return [...c, { role: 'assistant', content: data.content, speaker: data.speaker }]
      })
    })

    es.addEventListener('tool', (e) => {
      const data = JSON.parse((e as MessageEvent).data) as ToolEventData
      setTools((prev) => prev.some(t => t.id === data.id) ? prev : [...prev, data])
    })

    es.addEventListener('ask_user', (e) => {
      const data = JSON.parse((e as MessageEvent).data) as AskEvent
      setPendingAsk(data)
      if (data.current_agent) setCurrentAgent(data.current_agent)
      if (data.question?.trim()) {
        setChat((c) => {
          const last = c[c.length - 1]
          if (last && last.role === 'assistant' && last.content === data.question) return c
          return [...c, { role: 'assistant', content: data.question!, speaker: data.speaker || data.current_agent || 'AI' }]
        })
      }
      es.close()
    })

    es.addEventListener('final', (e) => {
      const data = JSON.parse((e as MessageEvent).data) as FinalEventData
      if (data.current_agent) setCurrentAgent(data.current_agent)
      if (data.message) setChat((c) => [...c, { role: 'assistant', content: data.message! }])
      es.close()
    })

    es.onerror = () => {
      es.close()
    }
  }, [token])

  const resumeStream = useCallback((tid: string, reply: string) => {
    const activeToken = token || localStorage.getItem('token')
    if (!activeToken) {
      setAuthError('Session expired. Please log in again.')
      return
    }
    const url = `${BACKEND}/api/graph/resume/stream?thread_id=${encodeURIComponent(tid)}&user_reply=${encodeURIComponent(reply)}&token=${encodeURIComponent(activeToken)}`
    const es = new EventSource(url)
  setPendingAsk(null)
  setChat((c) => [...c, { role: 'user', content: reply }])

    es.addEventListener('message', (e) => {
      const data = JSON.parse((e as MessageEvent).data) as MessageEventData
      if (data.current_agent) setCurrentAgent(data.current_agent)
      if (!data.content?.trim()) return // Do not render empty messages
      setChat((c) => {
        const last = c[c.length - 1]
        if (last && last.role === 'assistant' && last.content === data.content) return c
        return [...c, { role: 'assistant', content: data.content, speaker: data.speaker }]
      })
    })

    es.addEventListener('tool', (e) => {
      const data = JSON.parse((e as MessageEvent).data) as ToolEventData
      setTools((prev) => prev.some(t => t.id === data.id) ? prev : [...prev, data])
    })

    es.addEventListener('ask_user', (e) => {
      const data = JSON.parse((e as MessageEvent).data) as AskEvent
      setPendingAsk(data)
      if (data.current_agent) setCurrentAgent(data.current_agent)
      if (data.question?.trim()) {
        setChat((c) => {
          const last = c[c.length - 1]
          if (last && last.role === 'assistant' && last.content === data.question) return c
          return [...c, { role: 'assistant', content: data.question!, speaker: data.speaker || data.current_agent || 'AI' }]
        })
      }
      es.close()
    })

    es.addEventListener('final', (e) => {
      const data = JSON.parse((e as MessageEvent).data) as FinalEventData
      if (data.current_agent) setCurrentAgent(data.current_agent)
      if (data.message) setChat((c) => [...c, { role: 'assistant', content: data.message! }])
      es.close()
    })

    es.onerror = () => {
      es.close()
    }
  }, [token])

  const onSubmit = useCallback((e: FormEvent) => {
    e.preventDefault()
    const val = inputRef.current?.value?.trim()
    if (!val) return
    if (!(token || localStorage.getItem('token'))) {
      setAuthError('Please log in before chatting.')
      return
    }
    if (!threadId) {
      setChat((prev: ChatItem[]) => [...prev, { role: 'user', content: val }])
      startStream(val)
    } else if (pendingAsk?.thread_id === threadId) {
      resumeStream(threadId, val)
    }
    if (inputRef.current) inputRef.current.value = ''
  }, [threadId, pendingAsk, startStream, resumeStream, token])

  // Auto-scroll chat & tools
  useEffect(() => { if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight }, [chat])
  useEffect(() => { if (toolListRef.current) toolListRef.current.scrollTop = toolListRef.current.scrollHeight }, [tools])

  return (
    <div className="app-root">
      <div className="liquid-bg" />
      <div className="cursor-glow" />
      <div ref={containerRef} className="container glass tilt">
        <header className="header">
          <h1>AI Hospital</h1>
          <div className="status-tag glass">
            <div className="status-clip" />
            <div className="status-body glass-inner">
              <span className="status-label">Current Node</span>
              <span className="status-value">{currentAgent || '—'}</span>
            </div>
          </div>
          <div className="badge">Live</div>
        </header>
        <section className="auth-panel glass-inner">
          {!token ? (
            <form className="auth-form" onSubmit={handleLogin}>
              <h2>Patient Login</h2>
              <input
                type="email"
                value={email}
                placeholder="Email"
                onChange={(e) => setEmail(e.target.value)}
              />
              <input
                type="password"
                value={password}
                placeholder="Password"
                onChange={(e) => setPassword(e.target.value)}
              />
              <button className="btn" type="submit" disabled={authLoading}>
                {authLoading ? 'Signing in…' : 'Login'}
              </button>
            </form>
          ) : (
            <div className="auth-status">
              <span>Logged in</span>
              <button className="btn" type="button" onClick={handleLogout}>Logout</button>
            </div>
          )}
          {authError && <p className="auth-error">{authError}</p>}
        </section>
        <div className="main-row">
          <main className="chat glass-inner" ref={chatRef}>
            {chat.map((m: ChatItem, i: number) => (
              <div key={i} className={`bubble ${m.role === 'user' ? 'user' : 'assistant'}`}>
                {m.role === 'assistant' && (
                  <div className="speaker">{m.speaker || 'AI'}</div>
                )}
                <div className="content">{m.content}</div>
              </div>
            ))}
            {pendingAsk && (
              <div className="bubble assistant ask">
                <div className="speaker">{pendingAsk.speaker || 'Question'}</div>
                <div className="content">Waiting for your response…</div>
              </div>
            )}
          </main>
          <aside className="tool-panel glass-inner">
            <div className="tool-panel-header">Tool Calls</div>
            <div className="tool-list" ref={toolListRef}>
              {tools.map((t, idx) => (
                <div key={t.id} className={`tool-card tool-${t.name?.replace(/[^a-z0-9_]/gi,'').toLowerCase()}`}>
                  {idx > 0 && <div className="tool-arrow">↓</div>}
                  <div className="tool-name">{t.name}</div>
                  <div className="tool-meta">
                    <span className="tool-agent">{t.agent || 'Agent'}</span>
                    {t.args && Object.keys(t.args).length > 0 && (
                      <details className="tool-args"><summary>args</summary><pre>{JSON.stringify(t.args, null, 2)}</pre></details>
                    )}
                  </div>
                </div>
              ))}
              {!tools.length && <div className="tool-empty">Tool activity will appear here...</div>}
            </div>
          </aside>
        </div>
        <div className="bottom-block">
          <form onSubmit={onSubmit} className="composer glass-inner">
            <input ref={inputRef} className="input" type="text" placeholder={threadId ? 'Type your answer…' : 'Describe your issue…'} />
            <button className="btn" type="submit">{threadId ? (pendingAsk ? 'Answer' : 'Continue') : 'Start'}</button>
          </form>
          <footer className="footer">
            Backend base: {BACKEND || '(proxy /api → http://localhost:8000)'}
          </footer>
        </div>
      </div>
    </div>
  )
}
