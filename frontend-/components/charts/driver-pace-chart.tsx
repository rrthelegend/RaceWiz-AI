"use client"

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts"
import { mockPaceData } from "@/lib/api"

export function DriverPaceChart() {
  // Transform data for the chart - showing gap to leader
  const leaderTime = mockPaceData[0].averageLapTime
  const chartData = mockPaceData.map(d => ({
    driver: d.driverCode,
    gap: parseFloat((d.averageLapTime - leaderTime).toFixed(3)),
    color: d.teamColor,
    fullName: d.driver
  }))

  return (
    <div className="h-[300px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 10, right: 30, left: 40, bottom: 10 }}
        >
          <XAxis 
            type="number" 
            tickFormatter={(v) => `+${v.toFixed(1)}s`}
            stroke="hsl(var(--muted-foreground))"
            fontSize={11}
            axisLine={{ stroke: 'hsl(var(--border))' }}
            tickLine={{ stroke: 'hsl(var(--border))' }}
          />
          <YAxis 
            type="category" 
            dataKey="driver" 
            stroke="hsl(var(--muted-foreground))"
            fontSize={11}
            axisLine={{ stroke: 'hsl(var(--border))' }}
            tickLine={false}
            width={35}
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
              `+${value.toFixed(3)}s to leader`,
              props.payload.fullName
            ]}
            labelFormatter={() => ''}
          />
          <Bar 
            dataKey="gap" 
            radius={[0, 4, 4, 0]}
            maxBarSize={24}
          >
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
