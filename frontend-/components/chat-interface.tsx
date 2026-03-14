"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { Send, Bot, User, Sparkles, ChevronRight } from "lucide-react"

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
}

const suggestedQueries = [
  "Why did Red Bull dominate Monaco 2023?",
  "What was the best tyre strategy?",
  "Which driver had the best race pace?",
  "Compare Hamilton vs Verstappen lap times",
]

const mockResponses: Record<string, string> = {
  "red bull": `Red Bull's dominance at Monaco 2023 can be attributed to several key factors:

**1. Superior Mechanical Grip**
The RB19 excelled in the low-speed corners that define Monaco, with exceptional traction out of hairpins like the Grand Hotel and Rascasse.

**2. Optimal Setup**
Adrian Newey's team found the perfect balance between downforce and drag, crucial for a track where overtaking is nearly impossible.

**3. Verstappen's Qualifying Pace**
Max secured pole position with a 1:11.365, giving him track position which is paramount at Monaco.

**4. Strategic Execution**
Red Bull's pit strategy was flawless, with a well-timed stop during the Safety Car period that maintained track position.`,

  "tyre strategy": `**Optimal Tyre Strategy Analysis - Monaco GP 2023**

The best performing strategies were:

**2-Stop Medium-Hard-Soft** (Used by Verstappen)
- Start: Medium (Laps 1-22)
- Stint 2: Hard (Laps 23-54)
- Final: Soft (Laps 55-78) for fastest lap attempt

*Effectiveness: 94%*

**1-Stop Medium-Hard** (Used by Perez)
- Aggressive early stint preservation
- Longer hard compound run
- Less risky but slower overall

*Key Insight:* The 2-stop allowed fresher tyres for the final stint, enabling faster lap times without significant position loss due to Monaco's overtaking difficulty.`,

  "race pace": `**Race Pace Analysis - Top 5 Drivers**

| Driver | Avg Lap Time | Fastest Lap | Consistency |
|--------|-------------|-------------|-------------|
| VER | 1:12.432 | 1:11.891 | 98.5% |
| PER | 1:12.891 | 1:12.234 | 97.2% |
| HAM | 1:13.012 | 1:12.456 | 96.8% |
| RUS | 1:13.156 | 1:12.678 | 96.1% |
| SAI | 1:13.234 | 1:12.789 | 95.4% |

**Key Findings:**
- Verstappen's pace was **0.459s** faster per lap on average
- Hamilton showed strong consistency despite older Mercedes package
- Ferrari struggled with tyre degradation in the middle stint`,

  "hamilton": `**Hamilton vs Verstappen - Monaco GP 2023 Comparison**

**Qualifying:**
- VER: 1:11.365 (P1)
- HAM: 1:12.018 (P3)
- Gap: +0.653s

**Race Pace by Stint:**

*Stint 1 (Laps 1-22):*
- VER avg: 1:14.234
- HAM avg: 1:14.567
- Gap: +0.333s

*Stint 2 (Laps 23-50):*
- VER avg: 1:13.891
- HAM avg: 1:14.123
- Gap: +0.232s

*Final Stint:*
- VER avg: 1:12.678
- HAM avg: 1:13.012
- Gap: +0.334s

**Conclusion:** Verstappen maintained a consistent advantage throughout, with superior traction and mechanical grip giving him approximately 0.3s per lap advantage.`,
}

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isTyping, setIsTyping] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  const handleSend = async () => {
    if (!input.trim()) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInput("")
    setIsTyping(true)

    // Simulate AI response
    setTimeout(() => {
      const lowerInput = input.toLowerCase()
      let response = "I can help you analyze F1 race data, strategies, and driver performance. Try asking about specific races, tyre strategies, or driver comparisons!"
      
      for (const [key, value] of Object.entries(mockResponses)) {
        if (lowerInput.includes(key)) {
          response = value
          break
        }
      }

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response,
        timestamp: new Date()
      }

      setMessages(prev => [...prev, assistantMessage])
      setIsTyping(false)
    }, 1500)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleSuggestion = (query: string) => {
    setInput(query)
    textareaRef.current?.focus()
  }

  return (
    <div className="flex h-[calc(100vh-8rem)] flex-col rounded-lg border border-border bg-card">
      {/* Header */}
      <div className="flex items-center gap-3 border-b border-border p-4">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
          <Bot className="h-5 w-5 text-primary" />
        </div>
        <div>
          <h2 className="font-semibold text-foreground">RaceWiz AI Assistant</h2>
          <p className="text-xs text-muted-foreground">Powered by advanced race analytics</p>
        </div>
        <Badge variant="outline" className="ml-auto border-chart-2/50 text-chart-2">
          <Sparkles className="mr-1 h-3 w-3" />
          AI Enabled
        </Badge>
      </div>

      {/* Messages */}
      <ScrollArea className="flex-1 p-4" ref={scrollRef}>
        {messages.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center text-center">
            <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
              <Bot className="h-8 w-8 text-primary" />
            </div>
            <h3 className="mb-2 text-lg font-semibold text-foreground">
              Ask me about F1 strategy
            </h3>
            <p className="mb-6 max-w-md text-sm text-muted-foreground">
              I can analyze race data, explain tyre strategies, compare driver performances, and provide insights on team tactics.
            </p>
            <div className="grid gap-2 sm:grid-cols-2">
              {suggestedQueries.map((query, index) => (
                <button
                  key={index}
                  onClick={() => handleSuggestion(query)}
                  className="flex items-center gap-2 rounded-lg bg-secondary/50 px-4 py-2 text-left text-sm text-foreground transition-colors hover:bg-secondary"
                >
                  <ChevronRight className="h-4 w-4 text-muted-foreground" />
                  <span className="line-clamp-1">{query}</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-3 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                {message.role === 'assistant' && (
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/10">
                    <Bot className="h-4 w-4 text-primary" />
                  </div>
                )}
                <div
                  className={`max-w-[80%] rounded-lg px-4 py-3 ${
                    message.role === 'user'
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-secondary text-foreground'
                  }`}
                >
                  <div className="text-sm whitespace-pre-wrap leading-relaxed">
                    {message.content}
                  </div>
                  <div className={`mt-1 text-[10px] ${
                    message.role === 'user' ? 'text-primary-foreground/70' : 'text-muted-foreground'
                  }`}>
                    {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                </div>
                {message.role === 'user' && (
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-secondary">
                    <User className="h-4 w-4 text-muted-foreground" />
                  </div>
                )}
              </div>
            ))}
            {isTyping && (
              <div className="flex gap-3">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/10">
                  <Bot className="h-4 w-4 text-primary" />
                </div>
                <div className="rounded-lg bg-secondary px-4 py-3">
                  <div className="flex gap-1">
                    <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground [animation-delay:-0.3s]" />
                    <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground [animation-delay:-0.15s]" />
                    <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground" />
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </ScrollArea>

      {/* Input */}
      <div className="border-t border-border p-4">
        <div className="flex gap-2">
          <Textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about race strategy, tyre performance, or driver analysis..."
            className="min-h-[44px] max-h-32 resize-none bg-secondary/50 border-0"
            rows={1}
          />
          <Button 
            onClick={handleSend} 
            disabled={!input.trim() || isTyping}
            size="icon"
            className="h-11 w-11 shrink-0"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
        <p className="mt-2 text-center text-[10px] text-muted-foreground">
          RaceWiz AI may produce inaccurate information. Verify critical race decisions.
        </p>
      </div>
    </div>
  )
}
