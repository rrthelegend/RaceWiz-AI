"use client"

import { mockDrivers } from "@/lib/api"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"

export function DriversTable() {
  return (
    <div className="rounded-lg border border-border bg-card overflow-hidden">
      <Table>
        <TableHeader>
          <TableRow className="border-border hover:bg-transparent">
            <TableHead className="w-14 text-muted-foreground">Pos</TableHead>
            <TableHead className="text-muted-foreground">Driver</TableHead>
            <TableHead className="text-muted-foreground">Team</TableHead>
            <TableHead className="text-right text-muted-foreground">Best Lap</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {mockDrivers.map((driver, index) => (
            <TableRow 
              key={driver.id} 
              className="border-border hover:bg-secondary/50 transition-colors"
            >
              <TableCell className="font-mono">
                <div className="flex items-center gap-2">
                  {index === 0 && (
                    <div className="h-1.5 w-1.5 rounded-full bg-chart-3 animate-pulse" />
                  )}
                  <span className={index === 0 ? "text-primary font-bold" : "text-muted-foreground"}>
                    {driver.position}
                  </span>
                </div>
              </TableCell>
              <TableCell>
                <div className="flex items-center gap-3">
                  <div 
                    className="h-8 w-1 rounded-full"
                    style={{ backgroundColor: driver.teamColor }}
                  />
                  <div>
                    <div className="font-medium text-foreground">{driver.name}</div>
                    <div className="text-xs text-muted-foreground">#{driver.number}</div>
                  </div>
                </div>
              </TableCell>
              <TableCell>
                <Badge 
                  variant="outline" 
                  className="border-transparent"
                  style={{ 
                    backgroundColor: `${driver.teamColor}20`,
                    color: driver.teamColor
                  }}
                >
                  {driver.team}
                </Badge>
              </TableCell>
              <TableCell className="text-right">
                <span className={`font-mono text-sm ${index === 0 ? "text-chart-3 font-bold" : "text-foreground"}`}>
                  {driver.bestLapTime}
                </span>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}
