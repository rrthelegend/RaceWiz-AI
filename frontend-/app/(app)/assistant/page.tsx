import { Badge } from "@/components/ui/badge"
import { ChatInterface } from "@/components/chat-interface"
import { Card, CardContent } from "@/components/ui/card"
import { 
  MessageSquare, 
  Sparkles,
  Zap,
  Brain,
  Database
} from "lucide-react"

export default function AssistantPage() {
  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-foreground">
            AI Assistant
          </h1>
          <p className="text-sm text-muted-foreground">
            Ask questions about race strategy, telemetry, and performance analysis
          </p>
        </div>
        <Badge variant="outline" className="gap-1.5 border-primary/50 text-primary">
          <Sparkles className="h-3 w-3" />
          Powered by RaceWiz AI
        </Badge>
      </div>

      {/* Capabilities */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                <MessageSquare className="h-5 w-5 text-primary" />
              </div>
              <div>
                <p className="text-sm font-medium text-foreground">Natural Language</p>
                <p className="text-xs text-muted-foreground">Ask in plain English</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-chart-2/10">
                <Database className="h-5 w-5 text-chart-2" />
              </div>
              <div>
                <p className="text-sm font-medium text-foreground">Race Data</p>
                <p className="text-xs text-muted-foreground">2020-2024 seasons</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-chart-3/10">
                <Brain className="h-5 w-5 text-chart-3" />
              </div>
              <div>
                <p className="text-sm font-medium text-foreground">AI Analysis</p>
                <p className="text-xs text-muted-foreground">Deep insights</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-chart-4/10">
                <Zap className="h-5 w-5 text-chart-4" />
              </div>
              <div>
                <p className="text-sm font-medium text-foreground">Real-time</p>
                <p className="text-xs text-muted-foreground">Instant responses</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Chat Interface */}
      <ChatInterface />
    </div>
  )
}
