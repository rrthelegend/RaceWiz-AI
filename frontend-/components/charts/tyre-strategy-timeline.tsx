"use client"

import { mockTyreStrategies } from "@/lib/api"

const compoundColors: Record<string, { bg: string; text: string }> = {
  Soft: { bg: '#E8002D', text: '#FFFFFF' },
  Medium: { bg: '#FFC000', text: '#000000' },
  Hard: { bg: '#FFFFFF', text: '#000000' },
  Intermediate: { bg: '#43B02A', text: '#FFFFFF' },
  Wet: { bg: '#0080FF', text: '#FFFFFF' }
}

const totalLaps = 78

export function TyreStrategyTimeline() {
  return (
    <div className="space-y-3">
      {mockTyreStrategies.map((strategy) => (
        <div key={strategy.driverCode} className="flex items-center gap-3">
          <div className="w-10 text-xs font-mono font-bold text-foreground">
            {strategy.driverCode}
          </div>
          <div className="relative flex-1 h-7 rounded-md bg-secondary/30 overflow-hidden">
            {strategy.stints.map((stint, index) => {
              const startPercent = ((stint.startLap - 1) / totalLaps) * 100
              const widthPercent = ((stint.endLap - stint.startLap + 1) / totalLaps) * 100
              const colors = compoundColors[stint.compound]
              
              return (
                <div
                  key={index}
                  className="absolute top-0 h-full flex items-center justify-center text-[10px] font-bold transition-all duration-300 hover:brightness-110"
                  style={{
                    left: `${startPercent}%`,
                    width: `${widthPercent}%`,
                    backgroundColor: colors.bg,
                    color: colors.text,
                  }}
                >
                  <span className="truncate px-1">
                    {stint.compound.charAt(0)}
                  </span>
                </div>
              )
            })}
          </div>
          <div className="w-16 text-right text-[10px] text-muted-foreground">
            {strategy.stints.length} stops
          </div>
        </div>
      ))}
      
      {/* Legend */}
      <div className="flex items-center justify-center gap-4 pt-2 border-t border-border mt-4">
        {Object.entries(compoundColors).slice(0, 3).map(([compound, colors]) => (
          <div key={compound} className="flex items-center gap-1.5">
            <div 
              className="h-3 w-3 rounded-sm"
              style={{ backgroundColor: colors.bg }}
            />
            <span className="text-xs text-muted-foreground">{compound}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
