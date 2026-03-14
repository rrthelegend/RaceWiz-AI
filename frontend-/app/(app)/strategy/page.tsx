import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { StrategyPanel } from "@/components/strategy-panel"
import { TyreStrategyTimeline } from "@/components/charts/tyre-strategy-timeline"
import { 
  Gauge, 
  Lightbulb, 
  AlertTriangle,
  CheckCircle2,
  TrendingUp
} from "lucide-react"

export default function StrategyPage() {
  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-foreground">
            Strategy Simulator
          </h1>
          <p className="text-sm text-muted-foreground">
            Simulate and optimize race strategies with AI-powered predictions
          </p>
        </div>
        <Badge variant="outline" className="gap-1.5 border-chart-3/50 text-chart-3">
          <Lightbulb className="h-3 w-3" />
          AI Predictions Active
        </Badge>
      </div>

      {/* Tips Cards */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <div className="flex items-start gap-3">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-chart-2/10">
                <CheckCircle2 className="h-4 w-4 text-chart-2" />
              </div>
              <div>
                <p className="text-sm font-medium text-foreground">Optimal Window</p>
                <p className="text-xs text-muted-foreground">Lap 22-28 for first pit stop based on tyre degradation curves</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <div className="flex items-start gap-3">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-chart-3/10">
                <TrendingUp className="h-4 w-4 text-chart-3" />
              </div>
              <div>
                <p className="text-sm font-medium text-foreground">Track Position</p>
                <p className="text-xs text-muted-foreground">Consider undercut potential during pit windows</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <div className="flex items-start gap-3">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-destructive/10">
                <AlertTriangle className="h-4 w-4 text-destructive" />
              </div>
              <div>
                <p className="text-sm font-medium text-foreground">Weather Alert</p>
                <p className="text-xs text-muted-foreground">20% chance of rain in final stint - monitor conditions</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Strategy Simulator */}
        <div>
          <StrategyPanel />
        </div>

        {/* Reference Data */}
        <div className="space-y-6">
          {/* Current Strategies */}
          <Card className="border-border bg-card">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2 text-base font-semibold">
                  <Gauge className="h-4 w-4 text-primary" />
                  Reference Strategies
                </CardTitle>
                <Badge variant="secondary" className="text-xs">
                  Top 6 Drivers
                </Badge>
              </div>
            </CardHeader>
            <CardContent>
              <TyreStrategyTimeline />
            </CardContent>
          </Card>

          {/* Strategy Insights */}
          <Card className="border-border bg-card">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-base font-semibold">
                <Lightbulb className="h-4 w-4 text-chart-3" />
                AI Strategy Insights
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="rounded-lg bg-secondary/50 p-3">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm font-medium text-foreground">Soft → Hard</span>
                  <Badge variant="outline" className="text-xs border-chart-2/50 text-chart-2">Recommended</Badge>
                </div>
                <p className="text-xs text-muted-foreground">
                  Best for front runners. Maximize early pace advantage with softs, then protect position on hards.
                </p>
              </div>
              
              <div className="rounded-lg bg-secondary/50 p-3">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm font-medium text-foreground">Medium → Hard</span>
                  <Badge variant="outline" className="text-xs">Conservative</Badge>
                </div>
                <p className="text-xs text-muted-foreground">
                  Lower risk strategy. Good for midfield, prioritizing consistency over raw pace.
                </p>
              </div>
              
              <div className="rounded-lg bg-secondary/50 p-3">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm font-medium text-foreground">Soft → Medium → Soft</span>
                  <Badge variant="outline" className="text-xs border-chart-3/50 text-chart-3">Aggressive</Badge>
                </div>
                <p className="text-xs text-muted-foreground">
                  High risk, high reward. 2-stop strategy for maximum pace but requires flawless execution.
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
