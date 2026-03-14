// Mock data for F1 analytics
// In production, these would be actual API calls

export interface Driver {
  id: string
  name: string
  team: string
  teamColor: string
  position: number
  bestLapTime: string
  averagePace: number
  country: string
  number: number
}

export interface LapData {
  lapNumber: number
  driver: string
  lapTime: string
  sector1: string
  sector2: string
  sector3: string
  tyreCompound: 'Soft' | 'Medium' | 'Hard' | 'Intermediate' | 'Wet'
  position: number
}

export interface RaceSession {
  id: string
  year: number
  raceName: string
  circuit: string
  date: string
  laps: number
  weather: string
  country: string
}

export interface TyreStint {
  driver: string
  driverCode: string
  stints: {
    compound: 'Soft' | 'Medium' | 'Hard' | 'Intermediate' | 'Wet'
    startLap: number
    endLap: number
  }[]
}

export interface PaceData {
  driver: string
  driverCode: string
  teamColor: string
  averageLapTime: number
  fastestLap: number
}

export interface DegradationData {
  lap: number
  lapTime: number
  compound: string
}

// Mock drivers data
export const mockDrivers: Driver[] = [
  { id: '1', name: 'Max Verstappen', team: 'Red Bull Racing', teamColor: '#3671C6', position: 1, bestLapTime: '1:12.432', averagePace: 72.8, country: 'NED', number: 1 },
  { id: '2', name: 'Sergio Perez', team: 'Red Bull Racing', teamColor: '#3671C6', position: 2, bestLapTime: '1:12.891', averagePace: 73.2, country: 'MEX', number: 11 },
  { id: '3', name: 'Lewis Hamilton', team: 'Mercedes', teamColor: '#27F4D2', position: 3, bestLapTime: '1:13.012', averagePace: 73.4, country: 'GBR', number: 44 },
  { id: '4', name: 'George Russell', team: 'Mercedes', teamColor: '#27F4D2', position: 4, bestLapTime: '1:13.156', averagePace: 73.5, country: 'GBR', number: 63 },
  { id: '5', name: 'Carlos Sainz', team: 'Ferrari', teamColor: '#E8002D', position: 5, bestLapTime: '1:13.234', averagePace: 73.6, country: 'ESP', number: 55 },
  { id: '6', name: 'Charles Leclerc', team: 'Ferrari', teamColor: '#E8002D', position: 6, bestLapTime: '1:13.298', averagePace: 73.7, country: 'MON', number: 16 },
  { id: '7', name: 'Lando Norris', team: 'McLaren', teamColor: '#FF8000', position: 7, bestLapTime: '1:13.412', averagePace: 73.8, country: 'GBR', number: 4 },
  { id: '8', name: 'Oscar Piastri', team: 'McLaren', teamColor: '#FF8000', position: 8, bestLapTime: '1:13.501', averagePace: 73.9, country: 'AUS', number: 81 },
  { id: '9', name: 'Fernando Alonso', team: 'Aston Martin', teamColor: '#229971', position: 9, bestLapTime: '1:13.623', averagePace: 74.0, country: 'ESP', number: 14 },
  { id: '10', name: 'Lance Stroll', team: 'Aston Martin', teamColor: '#229971', position: 10, bestLapTime: '1:13.789', averagePace: 74.2, country: 'CAN', number: 18 },
]

// Mock sessions
export const mockSessions: RaceSession[] = [
  { id: '1', year: 2024, raceName: 'Monaco Grand Prix', circuit: 'Circuit de Monaco', date: '2024-05-26', laps: 78, weather: 'Sunny, 24°C', country: 'Monaco' },
  { id: '2', year: 2024, raceName: 'British Grand Prix', circuit: 'Silverstone Circuit', date: '2024-07-07', laps: 52, weather: 'Cloudy, 18°C', country: 'United Kingdom' },
  { id: '3', year: 2024, raceName: 'Italian Grand Prix', circuit: 'Autodromo Nazionale Monza', date: '2024-09-01', laps: 53, weather: 'Sunny, 28°C', country: 'Italy' },
  { id: '4', year: 2023, raceName: 'Abu Dhabi Grand Prix', circuit: 'Yas Marina Circuit', date: '2023-11-26', laps: 58, weather: 'Clear, 26°C', country: 'UAE' },
]

// Mock tyre strategies
export const mockTyreStrategies: TyreStint[] = [
  { driver: 'Max Verstappen', driverCode: 'VER', stints: [{ compound: 'Soft', startLap: 1, endLap: 18 }, { compound: 'Medium', startLap: 19, endLap: 42 }, { compound: 'Hard', startLap: 43, endLap: 78 }] },
  { driver: 'Sergio Perez', driverCode: 'PER', stints: [{ compound: 'Medium', startLap: 1, endLap: 25 }, { compound: 'Hard', startLap: 26, endLap: 78 }] },
  { driver: 'Lewis Hamilton', driverCode: 'HAM', stints: [{ compound: 'Medium', startLap: 1, endLap: 22 }, { compound: 'Hard', startLap: 23, endLap: 55 }, { compound: 'Soft', startLap: 56, endLap: 78 }] },
  { driver: 'George Russell', driverCode: 'RUS', stints: [{ compound: 'Soft', startLap: 1, endLap: 15 }, { compound: 'Medium', startLap: 16, endLap: 45 }, { compound: 'Medium', startLap: 46, endLap: 78 }] },
  { driver: 'Carlos Sainz', driverCode: 'SAI', stints: [{ compound: 'Hard', startLap: 1, endLap: 35 }, { compound: 'Medium', startLap: 36, endLap: 78 }] },
  { driver: 'Charles Leclerc', driverCode: 'LEC', stints: [{ compound: 'Soft', startLap: 1, endLap: 20 }, { compound: 'Hard', startLap: 21, endLap: 50 }, { compound: 'Soft', startLap: 51, endLap: 78 }] },
]

// Mock pace data for charts
export const mockPaceData: PaceData[] = [
  { driver: 'Max Verstappen', driverCode: 'VER', teamColor: '#3671C6', averageLapTime: 72.432, fastestLap: 71.891 },
  { driver: 'Sergio Perez', driverCode: 'PER', teamColor: '#3671C6', averageLapTime: 72.891, fastestLap: 72.234 },
  { driver: 'Lewis Hamilton', driverCode: 'HAM', teamColor: '#27F4D2', averageLapTime: 73.012, fastestLap: 72.456 },
  { driver: 'George Russell', driverCode: 'RUS', teamColor: '#27F4D2', averageLapTime: 73.156, fastestLap: 72.678 },
  { driver: 'Carlos Sainz', driverCode: 'SAI', teamColor: '#E8002D', averageLapTime: 73.234, fastestLap: 72.789 },
  { driver: 'Charles Leclerc', driverCode: 'LEC', teamColor: '#E8002D', averageLapTime: 73.298, fastestLap: 72.901 },
  { driver: 'Lando Norris', driverCode: 'NOR', teamColor: '#FF8000', averageLapTime: 73.412, fastestLap: 73.012 },
  { driver: 'Oscar Piastri', driverCode: 'PIA', teamColor: '#FF8000', averageLapTime: 73.501, fastestLap: 73.123 },
]

// Generate lap degradation data
export function generateDegradationData(driver: string): DegradationData[] {
  const baseTime = 72 + Math.random() * 2
  const data: DegradationData[] = []
  
  for (let lap = 1; lap <= 78; lap++) {
    let compound = 'Medium'
    let degradation = 0
    
    if (lap <= 18) {
      compound = 'Soft'
      degradation = lap * 0.02
    } else if (lap <= 42) {
      compound = 'Medium'
      degradation = (lap - 18) * 0.015
    } else {
      compound = 'Hard'
      degradation = (lap - 42) * 0.01
    }
    
    const lapTime = baseTime + degradation + (Math.random() * 0.3 - 0.15)
    
    data.push({
      lap,
      lapTime: parseFloat(lapTime.toFixed(3)),
      compound
    })
  }
  
  return data
}

// Generate lap data table
export function generateLapData(driverFilter?: string): LapData[] {
  const compounds: ('Soft' | 'Medium' | 'Hard')[] = ['Soft', 'Medium', 'Hard']
  const drivers = ['VER', 'PER', 'HAM', 'RUS', 'SAI', 'LEC', 'NOR', 'PIA']
  const data: LapData[] = []
  
  for (let lap = 1; lap <= 10; lap++) {
    for (const driver of drivers) {
      if (driverFilter && driver !== driverFilter) continue
      
      const baseTime = 72 + Math.random() * 3
      const s1 = 20 + Math.random() * 2
      const s2 = 28 + Math.random() * 2
      const s3 = baseTime - s1 - s2 + 24
      
      data.push({
        lapNumber: lap,
        driver,
        lapTime: formatLapTime(baseTime),
        sector1: formatSectorTime(s1),
        sector2: formatSectorTime(s2),
        sector3: formatSectorTime(s3),
        tyreCompound: compounds[Math.floor(Math.random() * 3)],
        position: Math.floor(Math.random() * 8) + 1
      })
    }
  }
  
  return data.sort((a, b) => a.lapNumber - b.lapNumber || a.position - b.position)
}

function formatLapTime(seconds: number): string {
  const mins = Math.floor(seconds / 60)
  const secs = (seconds % 60).toFixed(3)
  return `${mins}:${secs.padStart(6, '0')}`
}

function formatSectorTime(seconds: number): string {
  return seconds.toFixed(3)
}

// Simulated API calls
export async function fetchSessions(): Promise<RaceSession[]> {
  return new Promise(resolve => setTimeout(() => resolve(mockSessions), 100))
}

export async function fetchDrivers(): Promise<Driver[]> {
  return new Promise(resolve => setTimeout(() => resolve(mockDrivers), 100))
}

export async function fetchPaceData(): Promise<PaceData[]> {
  return new Promise(resolve => setTimeout(() => resolve(mockPaceData), 100))
}

export async function fetchTyreStrategies(): Promise<TyreStint[]> {
  return new Promise(resolve => setTimeout(() => resolve(mockTyreStrategies), 100))
}

export async function fetchLapData(driver?: string): Promise<LapData[]> {
  return new Promise(resolve => setTimeout(() => resolve(generateLapData(driver)), 100))
}

export async function fetchDegradationData(driver: string): Promise<DegradationData[]> {
  return new Promise(resolve => setTimeout(() => resolve(generateDegradationData(driver)), 100))
}
