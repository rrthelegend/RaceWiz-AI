"use client"

import { useState, useEffect } from "react"
import { generateLapData, type LapData } from "@/lib/api"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"

const compoundColors: Record<string, { bg: string; text: string }> = {
  Soft: { bg: '#E8002D', text: '#FFFFFF' },
  Medium: { bg: '#FFC000', text: '#000000' },
  Hard: { bg: '#FFFFFF', text: '#000000' },
}

export function LapDataTable() {
  const [data, setData] = useState<LapData[]>([])
  const [driverFilter, setDriverFilter] = useState("all")

  useEffect(() => {
    const filtered = driverFilter === "all" 
      ? generateLapData() 
      : generateLapData(driverFilter)
    setData(filtered)
  }, [driverFilter])

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-foreground">Lap Data</h3>
        <Select value={driverFilter} onValueChange={setDriverFilter}>
          <SelectTrigger className="w-32 h-8 bg-secondary/50 border-0 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Drivers</SelectItem>
            <SelectItem value="VER">Verstappen</SelectItem>
            <SelectItem value="HAM">Hamilton</SelectItem>
            <SelectItem value="LEC">Leclerc</SelectItem>
            <SelectItem value="NOR">Norris</SelectItem>
          </SelectContent>
        </Select>
      </div>
      
      <div className="rounded-lg border border-border bg-card overflow-hidden">
        <ScrollArea className="h-[400px]">
          <Table>
            <TableHeader className="sticky top-0 bg-card z-10">
              <TableRow className="border-border hover:bg-transparent">
                <TableHead className="w-16 text-xs text-muted-foreground">Lap</TableHead>
                <TableHead className="text-xs text-muted-foreground">Driver</TableHead>
                <TableHead className="text-xs text-muted-foreground">Lap Time</TableHead>
                <TableHead className="text-xs text-muted-foreground">S1</TableHead>
                <TableHead className="text-xs text-muted-foreground">S2</TableHead>
                <TableHead className="text-xs text-muted-foreground">S3</TableHead>
                <TableHead className="text-xs text-muted-foreground">Tyre</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.slice(0, 50).map((lap, index) => {
                const colors = compoundColors[lap.tyreCompound] || compoundColors.Hard
                return (
                  <TableRow 
                    key={`${lap.lapNumber}-${lap.driver}-${index}`}
                    className="border-border hover:bg-secondary/50 transition-colors"
                  >
                    <TableCell className="font-mono text-xs text-muted-foreground">
                      {lap.lapNumber}
                    </TableCell>
                    <TableCell className="font-mono text-xs font-bold text-foreground">
                      {lap.driver}
                    </TableCell>
                    <TableCell className="font-mono text-xs text-foreground">
                      {lap.lapTime}
                    </TableCell>
                    <TableCell className="font-mono text-xs text-muted-foreground">
                      {lap.sector1}
                    </TableCell>
                    <TableCell className="font-mono text-xs text-muted-foreground">
                      {lap.sector2}
                    </TableCell>
                    <TableCell className="font-mono text-xs text-muted-foreground">
                      {lap.sector3}
                    </TableCell>
                    <TableCell>
                      <Badge 
                        variant="outline" 
                        className="text-[10px] border-0 px-2 py-0.5"
                        style={{ 
                          backgroundColor: colors.bg,
                          color: colors.text
                        }}
                      >
                        {lap.tyreCompound.charAt(0)}
                      </Badge>
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </ScrollArea>
      </div>
    </div>
  )
}
