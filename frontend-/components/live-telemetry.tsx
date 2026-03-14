"use client"

import { useState, useEffect } from "react"
import { cn } from "@/lib/utils"

interface TelemetryData {
  speed: number
  rpm: number
  gear: number
  throttle: number
  brake: number
  drs: boolean
}

export function LiveTelemetry({ className }: { className?: string }) {
  const [data, setData] = useState<TelemetryData>({
    speed: 0,
    rpm: 0,
    gear: 1,
    throttle: 0,
    brake: 0,
    drs: false
  })

  useEffect(() => {
    const interval = setInterval(() => {
      setData({
        speed: Math.floor(280 + Math.random() * 50),
        rpm: Math.floor(10000 + Math.random() * 4000),
        gear: Math.floor(4 + Math.random() * 4),
        throttle: Math.floor(70 + Math.random() * 30),
        brake: Math.random() > 0.8 ? Math.floor(Math.random() * 100) : 0,
        drs: Math.random() > 0.7
      })
    }, 100)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className={cn("rounded-lg border border-border bg-card p-4", className)}>
      <div className="mb-3 flex items-center justify-between">
        <span className="text-xs font-medium text-muted-foreground">LIVE TELEMETRY</span>
        <div className="flex items-center gap-1.5">
          <div className="h-2 w-2 animate-pulse rounded-full bg-chart-2" />
          <span className="text-xs text-chart-2">STREAMING</span>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        {/* Speed */}
        <div className="text-center">
          <div className="font-mono text-3xl font-bold text-foreground">
            {data.speed}
          </div>
          <div className="text-[10px] text-muted-foreground">KM/H</div>
        </div>

        {/* Gear */}
        <div className="text-center">
          <div className="font-mono text-3xl font-bold text-primary">
            {data.gear}
          </div>
          <div className="text-[10px] text-muted-foreground">GEAR</div>
        </div>

        {/* RPM */}
        <div className="text-center">
          <div className="font-mono text-xl font-bold text-foreground">
            {data.rpm.toLocaleString()}
          </div>
          <div className="text-[10px] text-muted-foreground">RPM</div>
        </div>
      </div>

      {/* Pedals */}
      <div className="mt-4 space-y-2">
        <div className="flex items-center gap-2">
          <span className="w-16 text-xs text-muted-foreground">Throttle</span>
          <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-secondary">
            <div 
              className="absolute left-0 top-0 h-full rounded-full bg-chart-2 transition-all duration-100"
              style={{ width: `${data.throttle}%` }}
            />
          </div>
          <span className="w-10 text-right font-mono text-xs text-foreground">{data.throttle}%</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-16 text-xs text-muted-foreground">Brake</span>
          <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-secondary">
            <div 
              className="absolute left-0 top-0 h-full rounded-full bg-primary transition-all duration-100"
              style={{ width: `${data.brake}%` }}
            />
          </div>
          <span className="w-10 text-right font-mono text-xs text-foreground">{data.brake}%</span>
        </div>
      </div>

      {/* DRS */}
      <div className="mt-3 flex items-center justify-center">
        <div className={cn(
          "rounded px-3 py-1 text-xs font-bold transition-all",
          data.drs 
            ? "bg-chart-2 text-background animate-pulse" 
            : "bg-secondary text-muted-foreground"
        )}>
          DRS {data.drs ? "OPEN" : "CLOSED"}
        </div>
      </div>
    </div>
  )
}
