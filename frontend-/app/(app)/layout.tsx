import { AppSidebar } from "@/components/app-sidebar"
import { TopBar } from "@/components/top-bar"

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="min-h-screen bg-background">
      <AppSidebar />
      <div className="ml-64">
        <TopBar />
        <main className="p-6">
          {children}
        </main>
      </div>
    </div>
  )
}
