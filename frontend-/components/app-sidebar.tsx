"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { 
  LayoutDashboard, 
  Search, 
  Gauge, 
  MessageSquare,
  Zap,
  Flag
} from "lucide-react"

const navigation = [
  { name: "Dashboard", href: "/", icon: LayoutDashboard },
  { name: "Race Explorer", href: "/explorer", icon: Search },
  { name: "Strategy Simulator", href: "/strategy", icon: Gauge },
  { name: "AI Assistant", href: "/assistant", icon: MessageSquare },
]

export function AppSidebar() {
  const pathname = usePathname()

  return (
    <aside className="fixed left-0 top-0 z-40 h-screen w-64 border-r border-border bg-sidebar">
      <div className="flex h-full flex-col">
        {/* Logo */}
        <div className="flex h-16 items-center gap-3 border-b border-border px-6">
          <div className="relative flex h-10 w-10 items-center justify-center rounded-lg bg-primary">
            <Flag className="h-5 w-5 text-primary-foreground" />
            <div className="absolute -right-0.5 -top-0.5 h-2.5 w-2.5 animate-pulse rounded-full bg-chart-2" />
          </div>
          <div className="flex flex-col">
            <span className="text-lg font-bold tracking-tight text-foreground">
              RaceWiz
            </span>
            <span className="text-xs font-medium text-primary">AI</span>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 space-y-1 px-3 py-4">
          <div className="mb-2 px-3">
            <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Analytics
            </span>
          </div>
          {navigation.map((item) => {
            const isActive = pathname === item.href
            return (
              <Link
                key={item.name}
                href={item.href}
                className={cn(
                  "group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all duration-200",
                  isActive
                    ? "bg-primary/10 text-primary"
                    : "text-muted-foreground hover:bg-secondary hover:text-foreground"
                )}
              >
                <item.icon 
                  className={cn(
                    "h-5 w-5 transition-colors",
                    isActive ? "text-primary" : "text-muted-foreground group-hover:text-foreground"
                  )} 
                />
                {item.name}
                {isActive && (
                  <div className="ml-auto h-1.5 w-1.5 rounded-full bg-primary animate-pulse" />
                )}
              </Link>
            )
          })}
        </nav>

        {/* Status Footer */}
        <div className="border-t border-border p-4">
          <div className="rounded-lg bg-secondary/50 p-3">
            <div className="flex items-center gap-2">
              <Zap className="h-4 w-4 text-chart-3" />
              <span className="text-xs font-medium text-foreground">Live Data</span>
            </div>
            <div className="mt-2 flex items-center gap-2">
              <div className="h-2 w-2 animate-pulse rounded-full bg-chart-2" />
              <span className="text-xs text-muted-foreground">Connected to F1 API</span>
            </div>
          </div>
        </div>
      </div>
    </aside>
  )
}
