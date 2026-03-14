"use client"

import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { MapPin, Calendar, Flag, Cloud, Timer, Trophy } from "lucide-react"

interface RaceSummaryProps {
  trackName?: string
  raceName?: string
  date?: string
  laps?: number
  weather?: string
  country?: string
}

export function RaceSummaryCard({
  trackName = "Circuit de Monaco",
  raceName = "Monaco Grand Prix",
  date = "May 26, 2024",
  laps = 78,
  weather = "Sunny, 24°C",
  country = "Monaco"
}: RaceSummaryProps) {
  return (
    <Card className="relative overflow-hidden border-border bg-card">
      {/* Racing stripe accent */}
      <div className="absolute left-0 top-0 h-full w-1 bg-primary" />
      
      <CardContent className="p-5">
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <Badge variant="outline" className="border-primary/50 text-primary">
              Round 7
            </Badge>
            <h3 className="text-lg font-bold text-foreground">{raceName}</h3>
            <p className="text-sm text-muted-foreground">{trackName}</p>
          </div>
          <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
            <Trophy className="h-6 w-6 text-primary" />
          </div>
        </div>
        
        <div className="mt-5 grid grid-cols-2 gap-4">
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-md bg-secondary">
              <MapPin className="h-4 w-4 text-muted-foreground" />
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Location</p>
              <p className="text-sm font-medium text-foreground">{country}</p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-md bg-secondary">
              <Calendar className="h-4 w-4 text-muted-foreground" />
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Date</p>
              <p className="text-sm font-medium text-foreground">{date}</p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-md bg-secondary">
              <Flag className="h-4 w-4 text-muted-foreground" />
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Race Distance</p>
              <p className="text-sm font-medium text-foreground">{laps} Laps</p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-md bg-secondary">
              <Cloud className="h-4 w-4 text-muted-foreground" />
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Weather</p>
              <p className="text-sm font-medium text-foreground">{weather}</p>
            </div>
          </div>
        </div>

        {/* Track times */}
        <div className="mt-5 rounded-lg bg-secondary/50 p-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Timer className="h-4 w-4 text-primary" />
              <span className="text-xs font-medium text-muted-foreground">Fastest Lap</span>
            </div>
            <span className="font-mono text-sm font-bold text-chart-1">1:12.432</span>
          </div>
          <div className="mt-1 flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Max Verstappen - Lap 54</span>
            <Badge variant="secondary" className="text-xs">Track Record</Badge>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
