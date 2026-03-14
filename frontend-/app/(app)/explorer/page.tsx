import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { DriversTable } from "@/components/tables/drivers-table"
import { LapDataTable } from "@/components/tables/lap-data-table"
import { Search, Database, Filter, Download } from "lucide-react"
import { Button } from "@/components/ui/button"

export default function ExplorerPage() {
  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-foreground">
            Race Explorer
          </h1>
          <p className="text-sm text-muted-foreground">
            Explore raw race data and detailed telemetry
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" className="gap-2">
            <Filter className="h-4 w-4" />
            Filters
          </Button>
          <Button variant="outline" size="sm" className="gap-2">
            <Download className="h-4 w-4" />
            Export
          </Button>
        </div>
      </div>

      {/* Data Stats */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                <Database className="h-5 w-5 text-primary" />
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Total Entries</p>
                <p className="text-xl font-bold text-foreground">6,240</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-chart-2/10">
                <Search className="h-5 w-5 text-chart-2" />
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Drivers</p>
                <p className="text-xl font-bold text-foreground">20</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-chart-3/10">
                <Database className="h-5 w-5 text-chart-3" />
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Sessions</p>
                <p className="text-xl font-bold text-foreground">5</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-chart-4/10">
                <Database className="h-5 w-5 text-chart-4" />
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Data Points</p>
                <p className="text-xl font-bold text-foreground">1.2M</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Drivers Table */}
        <Card className="border-border bg-card">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2 text-base font-semibold">
                Driver Standings
              </CardTitle>
              <Badge variant="secondary" className="text-xs">
                Final Classification
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            <DriversTable />
          </CardContent>
        </Card>

        {/* Lap Data */}
        <Card className="border-border bg-card">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-base font-semibold">
              Lap Telemetry Data
            </CardTitle>
          </CardHeader>
          <CardContent>
            <LapDataTable />
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
