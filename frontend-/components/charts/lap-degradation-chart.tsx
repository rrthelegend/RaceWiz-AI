"use client"

import { useState, useEffect } from "react"
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { generateDegradationData, type DegradationData } from "@/lib/api"

const drivers = [
  { code: "VER", name: "Verstappen" },
  { code: "PER", name: "Perez" },
  { code: "HAM", name: "Hamilton" },
  { code: "RUS", name: "Russell" },
  { code: "SAI", name: "Sainz" },
  { code: "LEC", name: "Leclerc" },
]

const compoundColors: Record<string, string> = {
  Soft: '#E8002D',
  Medium: '#FFC000',
  Hard: '#FFFFFF',
  Intermediate: '#43B02A',
  Wet: '#0080FF'
}

export function LapDegradationChart() {
  const [selectedDriver, setSelectedDriver] = useState("VER")
  const [data, setData] = useState<DegradationData[]>([])

  useEffect(() => {
    setData(generateDegradationData(selectedDriver))
  }, [selectedDriver])

  const averageLapTime = data.length > 0 
    ? data.reduce((acc, d) => acc + d.lapTime, 0) / data.length 
    : 0

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <Select value={selectedDriver} onValueChange={setSelectedDriver}>
          <SelectTrigger className="w-40 bg-secondary/50 border-0">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {drivers.map(d => (
              <SelectItem key={d.code} value={d.code}>{d.name}</SelectItem>
            ))}
          </SelectContent>
        </Select>
        <div className="flex items-center gap-4 text-xs">
          {['Soft', 'Medium', 'Hard'].map(compound => (
            <div key={compound} className="flex items-center gap-1.5">
              <div 
                className="h-2.5 w-2.5 rounded-full" 
                style={{ backgroundColor: compoundColors[compound] }}
              />
              <span className="text-muted-foreground">{compound}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="h-[250px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 20, left: 10, bottom: 10 }}>
            <XAxis 
              dataKey="lap" 
              stroke="hsl(var(--muted-foreground))"
              fontSize={10}
              axisLine={{ stroke: 'hsl(var(--border))' }}
              tickLine={{ stroke: 'hsl(var(--border))' }}
              interval={9}
            />
            <YAxis 
              domain={['dataMin - 0.5', 'dataMax + 0.5']}
              stroke="hsl(var(--muted-foreground))"
              fontSize={10}
              axisLine={{ stroke: 'hsl(var(--border))' }}
              tickLine={{ stroke: 'hsl(var(--border))' }}
              tickFormatter={(v) => `${v.toFixed(1)}s`}
              width={45}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'hsl(var(--card))',
                border: '1px solid hsl(var(--border))',
                borderRadius: '8px',
                boxShadow: '0 4px 20px rgba(0,0,0,0.4)'
              }}
              labelStyle={{ color: 'hsl(var(--foreground))', fontWeight: 600 }}
              itemStyle={{ color: 'hsl(var(--muted-foreground))' }}
              formatter={(value: number, _name, props) => [
                `${value.toFixed(3)}s`,
                `Lap ${props.payload.lap} (${props.payload.compound})`
              ]}
              labelFormatter={(lap) => `Lap ${lap}`}
            />
            <ReferenceLine 
              y={averageLapTime} 
              stroke="hsl(var(--primary))" 
              strokeDasharray="4 4" 
              strokeOpacity={0.5}
            />
            <Line
              type="monotone"
              dataKey="lapTime"
              stroke="hsl(var(--chart-2))"
              strokeWidth={2}
              dot={(props) => {
                const { cx, cy, payload } = props
                return (
                  <circle
                    key={`dot-${payload.lap}`}
                    cx={cx}
                    cy={cy}
                    r={3}
                    fill={compoundColors[payload.compound]}
                    stroke="hsl(var(--background))"
                    strokeWidth={1}
                  />
                )
              }}
              activeDot={{
                r: 5,
                stroke: 'hsl(var(--primary))',
                strokeWidth: 2
              }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
