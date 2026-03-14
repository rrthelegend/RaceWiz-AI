"use client"

import { useState } from "react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Calendar, RefreshCw, Bell, Settings } from "lucide-react"

interface TopBarProps {
  onYearChange?: (year: string) => void
  onRaceChange?: (race: string) => void
  onSessionChange?: (session: string) => void
}

export function TopBar({ onYearChange, onRaceChange, onSessionChange }: TopBarProps) {
  const [year, setYear] = useState("2024")
  const [race, setRace] = useState("monaco")
  const [session, setSession] = useState("race")

  const handleYearChange = (value: string) => {
    setYear(value)
    onYearChange?.(value)
  }

  const handleRaceChange = (value: string) => {
    setRace(value)
    onRaceChange?.(value)
  }

  const handleSessionChange = (value: string) => {
    setSession(value)
    onSessionChange?.(value)
  }

  return (
    <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b border-border bg-background/95 px-6 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      {/* Left: Selectors */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 rounded-lg bg-secondary/50 px-3 py-1.5">
          <Calendar className="h-4 w-4 text-muted-foreground" />
          <Select value={year} onValueChange={handleYearChange}>
            <SelectTrigger className="h-7 w-20 border-0 bg-transparent p-0 text-sm font-medium focus:ring-0">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="2024">2024</SelectItem>
              <SelectItem value="2023">2023</SelectItem>
              <SelectItem value="2022">2022</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <Select value={race} onValueChange={handleRaceChange}>
          <SelectTrigger className="h-9 w-44 bg-secondary/50 border-0">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="monaco">Monaco GP</SelectItem>
            <SelectItem value="silverstone">British GP</SelectItem>
            <SelectItem value="monza">Italian GP</SelectItem>
            <SelectItem value="spa">Belgian GP</SelectItem>
            <SelectItem value="suzuka">Japanese GP</SelectItem>
          </SelectContent>
        </Select>

        <Select value={session} onValueChange={handleSessionChange}>
          <SelectTrigger className="h-9 w-32 bg-secondary/50 border-0">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="race">Race</SelectItem>
            <SelectItem value="qualifying">Qualifying</SelectItem>
            <SelectItem value="fp3">Practice 3</SelectItem>
            <SelectItem value="fp2">Practice 2</SelectItem>
            <SelectItem value="fp1">Practice 1</SelectItem>
          </SelectContent>
        </Select>

        <Badge variant="outline" className="border-primary/50 bg-primary/10 text-primary">
          <div className="mr-1.5 h-1.5 w-1.5 animate-pulse rounded-full bg-primary" />
          Live Session
        </Badge>
      </div>

      {/* Right: Actions */}
      <div className="flex items-center gap-2">
        <Button variant="ghost" size="icon" className="h-9 w-9 text-muted-foreground hover:text-foreground">
          <RefreshCw className="h-4 w-4" />
        </Button>
        <Button variant="ghost" size="icon" className="h-9 w-9 text-muted-foreground hover:text-foreground">
          <Bell className="h-4 w-4" />
        </Button>
        <Button variant="ghost" size="icon" className="h-9 w-9 text-muted-foreground hover:text-foreground">
          <Settings className="h-4 w-4" />
        </Button>
      </div>
    </header>
  )
}
