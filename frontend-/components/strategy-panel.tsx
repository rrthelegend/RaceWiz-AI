"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { 
  Play, 
  AlertTriangle, 
  CheckCircle2, 
  Clock, 
  Gauge,
  TrendingUp,
  Zap
} from "lucide-react"

const drivers = [
  { code: "VER", name: "Verstappen", team: "Red Bull Racing" },
  { code: "HAM", name: "Hamilton", team: "Mercedes" },
  { code: "LEC", name: "Leclerc", team: "Ferrari" },
  { code: "NOR", name: "Norris", team: "McLaren" },
]

const compounds = [
  { value: "soft", label: "Soft", color: "#E8002D" },
  { value: "medium", label: "Medium", color: "#FFC000" },
  { value: "hard", label: "Hard", color: "#FFFFFF" },
]

interface SimulationResult {
  estimatedTime: string
  degradationRisk: number
  pitEffectiveness: number
  recommendation: string
  riskLevel: 'low' | 'medium' | 'high'
}

export function StrategyPanel() {
  const [driver, setDriver] = useState("VER")
  const [compound, setCompound] = useState("medium")
  const [pitLap, setPitLap] = useState([25])
  const [isSimulating, setIsSimulating] = useState(false)
  const [result, setResult] = useState<SimulationResult | null>(null)

  const runSimulation = () => {
    setIsSimulating(true)
    setResult(null)
    
    // Simulate API call
    setTimeout(() => {
      const risk = compound === "soft" ? 75 : compound === "medium" ? 45 : 25
      const effectiveness = pitLap[0] < 30 ? 85 : pitLap[0] < 50 ? 70 : 55
      
      setResult({
        estimatedTime: `1:32:${Math.floor(Math.random() * 60).toString().padStart(2, '0')}.${Math.floor(Math.random() * 1000).toString().padStart(3, '0')}`,
        degradationRisk: risk + Math.floor(Math.random() * 20) - 10,
        pitEffectiveness: effectiveness + Math.floor(Math.random() * 10) - 5,
        recommendation: compound === "soft" 
          ? "High degradation expected. Consider earlier pit stop."
          : compound === "hard"
          ? "Conservative strategy. Good for track position."
          : "Balanced approach. Optimal for variable conditions.",
        riskLevel: risk > 60 ? 'high' : risk > 40 ? 'medium' : 'low'
      })
      setIsSimulating(false)
    }, 1500)
  }

  return (
    <div className="space-y-6">
      {/* Input Controls */}
      <Card className="border-border bg-card">
        <CardHeader className="pb-3">
          <CardTitle className="text-base font-semibold flex items-center gap-2">
            <Gauge className="h-4 w-4 text-primary" />
            Strategy Parameters
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-5">
          {/* Driver Selection */}
          <div className="space-y-2">
            <label className="text-xs font-medium text-muted-foreground">Driver</label>
            <Select value={driver} onValueChange={setDriver}>
              <SelectTrigger className="bg-secondary/50 border-0">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {drivers.map(d => (
                  <SelectItem key={d.code} value={d.code}>
                    <div className="flex items-center gap-2">
                      <span className="font-bold">{d.code}</span>
                      <span className="text-muted-foreground">- {d.team}</span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Compound Selection */}
          <div className="space-y-2">
            <label className="text-xs font-medium text-muted-foreground">Starting Compound</label>
            <div className="flex gap-2">
              {compounds.map(c => (
                <button
                  key={c.value}
                  onClick={() => setCompound(c.value)}
                  className={`flex-1 rounded-lg p-3 transition-all ${
                    compound === c.value 
                      ? 'ring-2 ring-primary bg-secondary' 
                      : 'bg-secondary/50 hover:bg-secondary'
                  }`}
                >
                  <div 
                    className="h-4 w-4 rounded-full mx-auto mb-1"
                    style={{ backgroundColor: c.color }}
                  />
                  <span className="text-xs font-medium text-foreground">{c.label}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Pit Lap Slider */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-xs font-medium text-muted-foreground">First Pit Stop Lap</label>
              <Badge variant="secondary" className="font-mono">Lap {pitLap[0]}</Badge>
            </div>
            <Slider
              value={pitLap}
              onValueChange={setPitLap}
              min={10}
              max={60}
              step={1}
              className="py-2"
            />
            <div className="flex justify-between text-[10px] text-muted-foreground">
              <span>Early (10)</span>
              <span>Mid (35)</span>
              <span>Late (60)</span>
            </div>
          </div>

          <Button 
            onClick={runSimulation} 
            className="w-full gap-2"
            disabled={isSimulating}
          >
            {isSimulating ? (
              <>
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary-foreground border-t-transparent" />
                Simulating...
              </>
            ) : (
              <>
                <Play className="h-4 w-4" />
                Run Simulation
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {/* Results */}
      {result && (
        <Card className="border-border bg-card animate-in fade-in slide-in-from-bottom-4 duration-500">
          <CardHeader className="pb-3">
            <CardTitle className="text-base font-semibold flex items-center gap-2">
              <Zap className="h-4 w-4 text-chart-3" />
              Simulation Results
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Estimated Race Time */}
            <div className="rounded-lg bg-secondary/50 p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Clock className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm text-muted-foreground">Estimated Race Time</span>
                </div>
                <span className="font-mono text-xl font-bold text-primary">{result.estimatedTime}</span>
              </div>
            </div>

            {/* Metrics */}
            <div className="grid gap-3">
              {/* Degradation Risk */}
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2">
                    {result.riskLevel === 'high' ? (
                      <AlertTriangle className="h-4 w-4 text-destructive" />
                    ) : (
                      <TrendingUp className="h-4 w-4 text-chart-2" />
                    )}
                    <span className="text-muted-foreground">Tyre Degradation Risk</span>
                  </div>
                  <span className={`font-medium ${
                    result.riskLevel === 'high' ? 'text-destructive' : 
                    result.riskLevel === 'medium' ? 'text-chart-3' : 'text-chart-2'
                  }`}>
                    {result.degradationRisk}%
                  </span>
                </div>
                <Progress 
                  value={result.degradationRisk} 
                  className="h-2"
                />
              </div>

              {/* Pit Effectiveness */}
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="h-4 w-4 text-chart-2" />
                    <span className="text-muted-foreground">Pit Stop Effectiveness</span>
                  </div>
                  <span className="font-medium text-chart-2">{result.pitEffectiveness}%</span>
                </div>
                <Progress 
                  value={result.pitEffectiveness} 
                  className="h-2"
                />
              </div>
            </div>

            {/* Recommendation */}
            <div className={`rounded-lg p-3 ${
              result.riskLevel === 'high' ? 'bg-destructive/10 border border-destructive/20' : 
              result.riskLevel === 'medium' ? 'bg-chart-3/10 border border-chart-3/20' : 
              'bg-chart-2/10 border border-chart-2/20'
            }`}>
              <p className="text-sm text-foreground">{result.recommendation}</p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Strategy Timeline Preview */}
      <Card className="border-border bg-card">
        <CardHeader className="pb-3">
          <CardTitle className="text-base font-semibold">Strategy Timeline</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="relative h-8 rounded-lg bg-secondary/30 overflow-hidden">
            {/* First stint */}
            <div 
              className="absolute left-0 top-0 h-full flex items-center justify-center text-[10px] font-bold"
              style={{ 
                width: `${(pitLap[0] / 78) * 100}%`,
                backgroundColor: compounds.find(c => c.value === compound)?.color,
                color: compound === 'hard' ? '#000' : '#fff'
              }}
            >
              Stint 1
            </div>
            {/* Second stint (assumed hard) */}
            <div 
              className="absolute top-0 h-full flex items-center justify-center text-[10px] font-bold"
              style={{ 
                left: `${(pitLap[0] / 78) * 100}%`,
                width: `${((78 - pitLap[0]) / 78) * 100}%`,
                backgroundColor: compound === 'hard' ? '#FFC000' : '#FFFFFF',
                color: '#000'
              }}
            >
              Stint 2
            </div>
          </div>
          <div className="mt-2 flex justify-between text-[10px] text-muted-foreground">
            <span>Start</span>
            <span>Lap {pitLap[0]} - Pit Stop</span>
            <span>Finish</span>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
