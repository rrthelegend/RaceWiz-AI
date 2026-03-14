import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { DriverPaceChart } from "@/components/charts/driver-pace-chart"
import { LapDegradationChart } from "@/components/charts/lap-degradation-chart"
import { TyreStrategyTimeline } from "@/components/charts/tyre-strategy-timeline"
import { RaceSummaryCard } from "@/components/race-summary-card"
import { LiveTelemetry } from "@/components/live-telemetry"
import { AppSidebar } from "@/components/app-sidebar"
import { TopBar } from "@/components/top-bar"
import { 
  Activity, 
  Timer, 
  Flag,
  TrendingUp,
  Gauge,
  Users
} from "lucide-react"

const standings = [
  { pos: 1, driver: 'VER', name: 'Verstappen', team: 'Red Bull', color: '#3671C6', gap: 'Leader' },
  { pos: 2, driver: 'PER', name: 'Perez', team: 'Red Bull', color: '#3671C6', gap: '+2.312s' },
  { pos: 3, driver: 'HAM', name: 'Hamilton', team: 'Mercedes', color: '#27F4D2', gap: '+5.891s' },
  { pos: 4, driver: 'RUS', name: 'Russell', team: 'Mercedes', color: '#27F4D2', gap: '+8.234s' },
  { pos: 5, driver: 'SAI', name: 'Sainz', team: 'Ferrari', color: '#E8002D', gap: '+12.567s' },
]

export default function DashboardPage() {
  return (
    <div className="min-h-screen bg-background">
      <AppSidebar />
      <div className="ml-64">
        <TopBar />
        <main className="p-6">
          <div className="space-y-6">
            {/* Page Header */}
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-2xl font-bold tracking-tight text-foreground">
                  Race Analytics Dashboard
                </h1>
                <p className="text-sm text-muted-foreground">
                  Real-time telemetry and performance insights
                </p>
              </div>
              <Badge variant="outline" className="gap-1.5 border-chart-2/50 text-chart-2">
                <Activity className="h-3 w-3" />
                Analyzing Monaco GP 2024
              </Badge>
            </div>

            {/* Quick Stats */}
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              <Card className="border-border bg-card">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs font-medium text-muted-foreground">Total Laps</p>
                      <p className="text-2xl font-bold text-foreground">78</p>
                    </div>
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                      <Flag className="h-5 w-5 text-primary" />
                    </div>
                  </div>
                  <p className="mt-2 text-xs text-muted-foreground">
                    <span className="text-chart-2">+2</span> from last year
                  </p>
                </CardContent>
              </Card>

              <Card className="border-border bg-card">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs font-medium text-muted-foreground">Fastest Lap</p>
                      <p className="text-2xl font-bold text-foreground font-mono">1:12.432</p>
                    </div>
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-chart-3/10">
                      <Timer className="h-5 w-5 text-chart-3" />
                    </div>
                  </div>
                  <p className="mt-2 text-xs text-muted-foreground">
                    VER - Lap 54
                  </p>
                </CardContent>
              </Card>

              <Card className="border-border bg-card">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs font-medium text-muted-foreground">Avg Speed</p>
                      <p className="text-2xl font-bold text-foreground">162.4 <span className="text-sm text-muted-foreground">km/h</span></p>
                    </div>
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-chart-2/10">
                      <Gauge className="h-5 w-5 text-chart-2" />
                    </div>
                  </div>
                  <p className="mt-2 text-xs text-muted-foreground">
                    <span className="text-primary">+1.2%</span> vs qualifying
                  </p>
                </CardContent>
              </Card>

              <Card className="border-border bg-card">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs font-medium text-muted-foreground">Overtakes</p>
                      <p className="text-2xl font-bold text-foreground">23</p>
                    </div>
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-chart-4/10">
                      <TrendingUp className="h-5 w-5 text-chart-4" />
                    </div>
                  </div>
                  <p className="mt-2 text-xs text-muted-foreground">
                    High action race
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* Main Dashboard Grid */}
            <div className="grid gap-6 lg:grid-cols-3">
              {/* Left Column - Charts */}
              <div className="space-y-6 lg:col-span-2">
                {/* Driver Pace Ranking */}
                <Card className="border-border bg-card">
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <CardTitle className="flex items-center gap-2 text-base font-semibold">
                        <Users className="h-4 w-4 text-primary" />
                        Driver Pace Ranking
                      </CardTitle>
                      <Badge variant="secondary" className="text-xs">
                        Gap to Leader
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <DriverPaceChart />
                  </CardContent>
                </Card>

                {/* Lap Time Degradation */}
                <Card className="border-border bg-card">
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <CardTitle className="flex items-center gap-2 text-base font-semibold">
                        <Activity className="h-4 w-4 text-chart-2" />
                        Lap Time Degradation
                      </CardTitle>
                      <Badge variant="secondary" className="text-xs">
                        By Stint
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <LapDegradationChart />
                  </CardContent>
                </Card>

                {/* Tyre Strategy */}
                <Card className="border-border bg-card">
                  <CardHeader className="pb-2">
                    <CardTitle className="flex items-center gap-2 text-base font-semibold">
                      <Gauge className="h-4 w-4 text-chart-3" />
                      Tyre Strategy Comparison
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <TyreStrategyTimeline />
                  </CardContent>
                </Card>
              </div>

              {/* Right Column - Race Summary */}
              <div className="space-y-6">
                <LiveTelemetry />
                <RaceSummaryCard />
                
                {/* Live Positions */}
                <Card className="border-border bg-card">
                  <CardHeader className="pb-2">
                    <CardTitle className="flex items-center gap-2 text-base font-semibold">
                      <Flag className="h-4 w-4 text-primary" />
                      Final Classification
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    {standings.map((driver) => (
                      <div
                        key={driver.pos}
                        className="flex items-center gap-3 rounded-lg bg-secondary/50 p-2 transition-colors hover:bg-secondary"
                      >
                        <div className={`flex h-6 w-6 items-center justify-center rounded text-xs font-bold ${
                          driver.pos === 1 ? 'bg-chart-3 text-background' : 'bg-muted text-muted-foreground'
                        }`}>
                          {driver.pos}
                        </div>
                        <div
                          className="h-6 w-1 rounded-full"
                          style={{ backgroundColor: driver.color }}
                        />
                        <div className="flex-1">
                          <p className="text-sm font-medium text-foreground">{driver.name}</p>
                          <p className="text-xs text-muted-foreground">{driver.team}</p>
                        </div>
                        <span className={`text-xs font-mono ${
                          driver.pos === 1 ? 'text-chart-2 font-bold' : 'text-muted-foreground'
                        }`}>
                          {driver.gap}
                        </span>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
