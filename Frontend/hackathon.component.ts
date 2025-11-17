import { ChangeDetectorRef, Component, OnInit, OnDestroy } from '@angular/core';
import { TempServService, StintSimulation, sendStint, getVehiclePerRaceSession, getLapNumberPerRaceSessionAndVehicle, LapNumbersResponse, getVehiclePerRaceSessionAndMinTime, PredictNewRaceRequest } from '../temp-serv.service';
import { firstValueFrom } from 'rxjs';
import { Chart, ChartConfiguration, registerables } from 'chart.js';
Chart.register(...registerables);

@Component({
  selector: 'app-hackathon',
  templateUrl: './hackathon.component.html',
  styleUrls: ['./hackathon.component.css']
})
export class HackathonComponent implements OnInit {
  private charts: Record<string, Chart> = {};
  private readonly MIN_LAP_TIME = 25;

  // Dropdown options
  Math = Math;
  tracks = ['Barber', 'Circuit of the Americas', 'Road America', 'Sonoma', 'Virginia International Raceway'];
  raceSessions = ['Race 1', 'Race 2'];
  vehicles: string[] = [];

  //model components
  trackAnalytics: Record<string, { r2: number; rmse: number, trackLength: number, avgPitStopTime: number }> = {
    'Barber': { r2: 96.929, rmse: 1.18, trackLength: 2.28, avgPitStopTime: 34 },
    'Circuit of the Americas': { r2: 86.88, rmse: 9.70, trackLength: 3.416, avgPitStopTime: 36 },
    'Road America': { r2: 95.621, rmse: 1.48, trackLength: 4.014, avgPitStopTime: 52 },
    'Sonoma': { r2: 80.01, rmse: 5.24, trackLength: 2.505, avgPitStopTime: 45 },
    'Virginia International Raceway': { r2: 98.15, rmse: 1.52, trackLength: 3.270, avgPitStopTime: 25 }
  };

  trackImages: Record<string, string> = {
    'Barber': 'assets/tracks/track_image_barber.png',
    'Circuit of the Americas': 'assets/tracks/track_image_cota.png',
    'Road America': 'assets/tracks/track_image_road_america.png',
    'Sonoma': 'assets/tracks/track_image_sonoma.png',
    'Virginia International Raceway': 'assets/tracks/track_image_vir.png'
  };

  currentTrackImage: string | null = null;

  // User selections
  selectedTrack: string | null = null;
  selectedSession: string | null = null;
  selectedSessionToSend: string | null = null;
  selectedVehicle: string | null = null;
  startingLap: number | null = null;
  startingLapSelected: number | null = null;
  stintLength: number | null = null;
  currentAnalytics: { r2: number; rmse: number, trackLength: number, avgPitStopTime: number } | null = null;
  isPitStop = false;
  helpingUserOutVehicle = "Loading vehicles...";
  howManyStartingLaps: LapNumbersResponse = { lap_numbers: [] };
  minLapOptions = Array.from({ length: 11 }, (_, i) => i * 10); // [0,10,...100]
  selectedMinLapTime = 60;

  newSessionOpen = false;
  newSessionForm: PredictNewRaceRequest = {
    track_name: 'Barber',
    vehicle_id: '',
    total_laps_to_predict: 10,
    previous_laps: [] as { lap:number, lap_time_seconds:number, laps_on_tires?:number, fuel_load_proxy?:number, session_air_temp?:number, session_track_temp?:number }[]
  };
  newSessionResult: any = null;

  // API sending object
  sendStintObj: sendStint = { track_name: "Barber", is_pit_stop: false, race_session: "R2", vehicle_id: "GR86-002-000", start_lap: 5, stint_length: 10 };
  sendTrackAndRaceSession: getVehiclePerRaceSession = { track_name: "Barber", race_session: "R2" };
  sendTrackAndRaceSessionAndVehicleId: getLapNumberPerRaceSessionAndVehicle = { track_name: "Barber", race_session: "R2", vehicle_id: "GR86-002-000" };
  sendTrackAndRaceSessionAndMinTime: getVehiclePerRaceSessionAndMinTime = { track_name: "Barber", race_session: "R2", min_lap_time_enforced: 60 };

  // Results
  stintResults: StintSimulation = { vehicle_id: '', predicted_lap_times: [], true_lap_times: [], race_session: '', start_lap: 0 };
  realTimeResults: any = null;
  postEventResults: any = null;
  isLoading = false;
  errorMessage = '';
  activeMode: 'pre' | 'real' | 'post' | 'def' | 'trueEvent' | 'newSession' | null = null;

  constructor(private predictionService: TempServService, private cd: ChangeDetectorRef) {}

  ngOnInit() {}

  async onTrackChange() {
    //Clear vehicles and sessions if track is changed
    this.selectedSession = null;
    this.vehicles = [];
    this.selectedVehicle = null;
    this.currentAnalytics = null;
    this.startingLap = null;
    this.howManyStartingLaps = { lap_numbers: [] };

    if (this.selectedTrack) {
      this.currentAnalytics = this.trackAnalytics[this.selectedTrack] || null;
      this.currentTrackImage = this.trackImages[this.selectedTrack] || null;
    } else {
      this.currentAnalytics = null;
      this.currentTrackImage = null;
    }
  }

  async onSessionChange() {
    // Update analytics panel if track is selected
    if (this.selectedTrack) {
      this.currentAnalytics = this.trackAnalytics[this.selectedTrack] || null;
    } else {
      this.currentAnalytics = null;
    }

    // Clear vehicles if either field is missing
    if (!this.selectedTrack || !this.selectedSession) {
      this.vehicles = [];
      this.selectedVehicle = null;
      return;
    }else{
      this.vehicles = [];
      this.selectedVehicle = null;
      this.startingLap = null;
      this.howManyStartingLaps = { lap_numbers: [] };
    }

    try {
      // --- Map user-facing race sessions to API format ---
      const sessionCode = this.selectedSession === 'Race 1' ? 'R1' : 'R2';

      this.sendTrackAndRaceSession = {
        track_name: this.selectedTrack,
        race_session: sessionCode
      };

      this.isLoading = true;

      // --- Fetch vehicles from Flask API ---
      const response = await firstValueFrom(
        this.predictionService.getVehicleIdsByRaceSession(this.sendTrackAndRaceSession)
      );

      if (response && response.vehicle_ids && response.vehicle_ids.length > 0) {
        this.vehicles = response.vehicle_ids;
        this.helpingUserOutVehicle = "Select Vehicle";
      } else {
        // fallback if API returns empty
        this.vehicles = [];
        this.selectedVehicle = null;
        console.warn(`No vehicles found for ${sessionCode}`);
      }

      this.cd.detectChanges();
    } catch (error) {
      this.errorMessage = 'Failed to load vehicle IDs.';
      console.error('Failed to load vehicle IDs:', error);
      // fallback list if API fails
      this.vehicles = ['Car-A', 'Car-B', 'Car-C'];
      this.helpingUserOutVehicle = "Select Vehicle";
    } finally {
      this.isLoading = false;
    }
  }

  async onVehicleChange() {
    if (!this.selectedTrack || !this.selectedSession || !this.selectedVehicle) {
      return;
    }

    try {
      const sessionCode = this.selectedSession === 'Race 1' ? 'R1' : 'R2';

      this.sendTrackAndRaceSessionAndVehicleId = {
        track_name: this.selectedTrack,
        race_session: sessionCode,
        vehicle_id: this.selectedVehicle
      };

      this.isLoading = true;

      const response = await firstValueFrom(
        this.predictionService.getLapNumbersByRaceSessionAndVehicleId(this.sendTrackAndRaceSessionAndVehicleId)
      );

      if (response && response.lap_numbers && response.lap_numbers.length > 0) {
        if (!response.lap_numbers.includes(1)) {
          response.lap_numbers.unshift(1);
        }
        this.howManyStartingLaps = response;
        // if (this.startingLap == null){
        this.startingLap = this.howManyStartingLaps.lap_numbers[0]; // auto-select first lap
        // }
      } else {
        console.warn(`No lap numbers found for ${this.selectedVehicle} in ${sessionCode}`);
      }
    } catch (error) {
      this.errorMessage = 'Failed to load lap numbers.';
      console.error('Failed to load lap numbers:', error);
    } finally {
      this.isLoading = false;
    }
  }

  isValidLapTime(time: number): boolean {
    return time !== null && time !== undefined && time >= this.MIN_LAP_TIME;
  }

  async onPredictStint() {
    this.errorMessage = '';
    this.stintResults = {} as StintSimulation;

    if (!this.selectedSession || !this.selectedVehicle || !this.startingLap || !this.stintLength) {
      this.errorMessage = 'Please fill all required fields.';
      return;
    }

    this.selectedSessionToSend = this.selectedSession === "Race 1" ? "R1" : "R2";
    this.startingLapSelected = this.startingLap;
    this.isLoading = true;

    this.sendStintObj = {
      track_name: this.selectedTrack || "Barber",
      is_pit_stop: this.isPitStop,
      race_session: this.selectedSessionToSend,
      vehicle_id: this.selectedVehicle,
      start_lap: this.startingLap,
      stint_length: this.stintLength
    };

    try {
      const result = await firstValueFrom(this.predictionService.simulateStint(this.sendStintObj));
      this.stintResults = result;
      if (this.activeMode != 'def') {
        this.cd.detectChanges();
        setTimeout(() => this.plotChart('preChart', result.predicted_lap_times, 'Pre-Event Predictions'), 0);
      }else{
        this.destroyAllCharts();
      }
    } catch (err) {
      console.error('Prediction failed:', err);
      this.errorMessage = 'Prediction failed. Please refresh the page to try again.';
    } finally {
      this.isLoading = false;
    }
  }

  async runDefault() {
    this.activeMode = 'def';
    await this.onPredictStint();
  }

  // --- Analytics Buttons ---
  async runPreEvent() {
    this.activeMode = 'pre';
    this.newSessionOpen = false;
    this.newSessionResult = null;
    await this.onPredictStint();
  }

  async runRealTime() {
    this.errorMessage = '';
    this.activeMode = 'real';
    this.newSessionOpen = false;
    this.newSessionResult = null;
    if (!this.selectedVehicle || !this.startingLap || !this.stintLength){
      this.errorMessage = 'Please fill all required fields.';
      return;
    }

    this.startingLapSelected = this.startingLap;
    const noPitObj = { ...this.sendStintObj, is_pit_stop: false };
    const pitObj = { ...this.sendStintObj, is_pit_stop: true };

    this.isLoading = true;
    const [noPit, pit] = await Promise.all([
      firstValueFrom(this.predictionService.simulateStint(noPitObj)),
      firstValueFrom(this.predictionService.simulateStint(pitObj))
    ]);
    this.realTimeResults = { noPit, pit };
    this.isLoading = false;

    this.cd.detectChanges();
    setTimeout(() => this.plotRealTimeChart(noPit.predicted_lap_times, pit.predicted_lap_times), 0);
  }

  async runPostEvent() {
    this.errorMessage = '';
    this.activeMode = 'post';
    this.newSessionOpen = false;
    this.newSessionResult = null;
    if (!this.stintResults.predicted_lap_times || !this.stintResults.true_lap_times || !this.stintResults.predicted_lap_times.length || !this.stintResults.true_lap_times.length){
      this.errorMessage = 'Please run a prediction first to get post-event analysis.';
      this.activeMode = null;
      return; 
    }

    this.startingLapSelected = this.startingLap;

    const actual = this.stintResults.true_lap_times;
    const predicted = this.stintResults.predicted_lap_times;

    const validPairs = actual.map((a, i) => ({ actual: a, predicted: predicted[i] }))
    .filter(pair => this.isValidLapTime(pair.actual) && this.isValidLapTime(pair.predicted));


    // --- Compute metrics ---
    const absErrors = validPairs.map(pair => Math.abs(pair.predicted - pair.actual));
    const mae = validPairs.length ? absErrors.reduce((a, b) => a + b, 0) / absErrors.length : 0;
    const rmse = validPairs.length ? 
      Math.sqrt(absErrors.map(e => e ** 2).reduce((a, b) => a + b, 0) / absErrors.length) : 0;

    // --- Compute trendline (simple linear regression) ---
    const laps = actual.map((_, i) => i + 1);
    const { m, b } = this.computeTrendline(laps, actual);
    const trendline = laps.map(x => m * x + b);

    // --- Pull model-level analytics ---
    const modelStats = this.currentAnalytics || { r2: 0, rmse: 0, trackLength: 0 };

    // --- Performance delta (compare to overall RMSE) ---
    const delta = rmse - modelStats.rmse;
    let performanceStatus = 'on par';
    if (delta <= -0.2) performanceStatus = 'above average';
    else if (delta >= 0.2) performanceStatus = 'below average';

    this.postEventResults = {
      actual,
      predicted,
      trendline,
      mae,
      rmse,
      model_r2: modelStats.r2,
      model_rmse: modelStats.rmse,
      performanceStatus,
      insight: this.getPostEventInsight(mae, rmse, modelStats.r2)
    };

    this.cd.detectChanges();
    setTimeout(() => this.plotPostEventChart(actual, predicted, trendline), 0);
  }

  async runTrueEvent() {
    this.errorMessage = '';
    this.activeMode = 'trueEvent';
    this.newSessionOpen = false;
    this.newSessionResult = null;

    if (!this.selectedTrack || !this.selectedSession) {
      this.vehicles = [];
      this.selectedVehicle = null;
      this.errorMessage = 'Please select track and session first.';
      return;
    }

    try {
      const sessionCode = this.selectedSession === 'Race 1' ? 'R1' : 'R2';

      this.sendTrackAndRaceSessionAndMinTime = {
        track_name: this.selectedTrack,
        race_session: sessionCode,
        min_lap_time_enforced: this.selectedMinLapTime
      };

      this.isLoading = true;

      const response = await firstValueFrom(
        this.predictionService.getFinalResultsByTrackAndRaceSession(this.sendTrackAndRaceSessionAndMinTime)
      );

      if (response && response.results?.length > 0) {
        this.postEventResults = response;   // ← SAVE RESULTS
      } else {
        this.postEventResults = null;
        console.warn(`No final results for ${sessionCode}`);
      }

      this.cd.detectChanges();

    } catch (error) {
      this.errorMessage = 'Failed to load final results.';
      console.error('Failed to load final results:', error);
    } finally {
      this.isLoading = false;
    }
  }

  private filterValidLapTimes(lapTimes: number[]): number[] {
    return lapTimes.filter(t => this.isValidLapTime(t));
  }

  // --- Chart plotting functions ---
  plotChart(elementId: string, data: number[], label: string) {
    const ctx = document.getElementById(elementId) as HTMLCanvasElement;
    if (!ctx) return;

    // Destroy any old chart
    if (this.charts[elementId]) this.charts[elementId].destroy();

    const trueTimes = this.stintResults.true_lap_times || [];
    const length = Math.max(data.length, trueTimes.length);
    const lapLabels = Array.from({ length }, (_, i) => i + 1);

    // Build plotting arrays preserving lap indexes. For invalid true times, substitute predicted value if valid.
    const plotPredicted = Array.from({ length }, (_, i) => {
      const p = i < data.length ? data[i] : null;
      return (p !== null && p !== undefined && this.isValidLapTime(p)) ? p : null;
    });

    const plotTrue = Array.from({ length }, (_, i) => {
      const t = i < trueTimes.length ? trueTimes[i] : null;
      if (t !== null && t !== undefined && this.isValidLapTime(t)) return t;
      // fallback to predicted value for plotting only
      const p = i < data.length ? data[i] : null;
      return (p !== null && p !== undefined && this.isValidLapTime(p)) ? p : null;
    });

    this.charts[elementId] = new Chart(ctx, {
      type: 'line',
      data: {
        labels: lapLabels,
        datasets: [
          { label: 'Predicted Lap Times', data: plotPredicted, borderColor: '#00BFFF', fill: false, tension: 0.2, spanGaps: true },
          { label: 'True Lap Times', data: plotTrue, borderColor: '#4CAF50', fill: false, tension: 0.2, spanGaps: true }
        ]
      },
      options: {
        responsive: true,
        plugins: { legend: { position: 'top' } },
        scales: {
          y: { title: { display: true, text: 'Lap Time (s)' } },
          x: { title: { display: true, text: 'Lap #' } }
        }
      }
    });
  }

  plotRealTimeChart(noPitData: number[], pitData: number[]) {
    const ctx = document.getElementById('realChart') as HTMLCanvasElement;
    if (!ctx) return;
    if (this.charts['realChart']) this.charts['realChart'].destroy();

    const validNoPit = this.filterValidLapTimes(noPitData);
    const validPit = this.filterValidLapTimes(pitData);
    const lapLabels = Array.from({ length: Math.max(validNoPit.length, validPit.length) }, (_, i) => i + 1);

    this.charts['realChart'] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: lapLabels,
        datasets: [
          { label: 'No Pit', data: validNoPit, backgroundColor: '#007bff' },
          { label: 'Pit Stop', data: validPit, backgroundColor: '#ff9800' }
        ]
      },
      options: {
        responsive: true,
        scales: {
          y: { beginAtZero: false, title: { display: true, text: 'Lap Time (s)' } },
          x: { title: { display: true, text: 'Lap #' } }
        }
      }
    });
  }

  plotPostEventChart(actual: number[], predicted: number[], trendline: number[]) {
    const ctx = document.getElementById('postChart') as HTMLCanvasElement;
    if (!ctx) return;
    if (this.charts['postChart']) this.charts['postChart'].destroy();

    const length = Math.max(actual.length, predicted.length, trendline.length);
    const lapLabels = Array.from({ length }, (_, i) => i + 1);

    // For plotting: if actual invalid, use predicted value for the true series (visual only)
    const plotActual = Array.from({ length }, (_, i) => {
      const a = i < actual.length ? actual[i] : null;
      if (a !== null && a !== undefined && this.isValidLapTime(a)) return a;
      const p = i < predicted.length ? predicted[i] : null;
      return (p !== null && p !== undefined && this.isValidLapTime(p)) ? p : null;
    });

    const plotPredicted = Array.from({ length }, (_, i) => {
      const p = i < predicted.length ? predicted[i] : null;
      return (p !== null && p !== undefined && this.isValidLapTime(p)) ? p : null;
    });

    // trendline only where actual was valid (optional) - keep nulls where actual invalid so trend doesn't mislead
    const plotTrend = Array.from({ length }, (_, i) => {
      const a = i < actual.length ? actual[i] : null;
      return (a !== null && a !== undefined && this.isValidLapTime(a) && i < trendline.length) ? trendline[i] : null;
    });

    this.charts['postChart'] = new Chart(ctx, {
      type: 'line',
      data: {
        labels: lapLabels,
        datasets: [
          { label: 'True Lap Times (visual fallback)', data: plotActual, borderColor: '#4caf50', fill: false, tension: 0.2, spanGaps: true },
          { label: 'Predicted Lap Times', data: plotPredicted, borderColor: '#00BFFF', fill: false, tension: 0.2, borderDash: [5, 5], spanGaps: true },
          { label: 'Trendline (True)', data: plotTrend, borderColor: '#ff9800', fill: false, borderDash: [3, 3], borderWidth: 1.5, pointRadius: 0, spanGaps: true }
        ]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { position: 'bottom' },
          title: { display: true, text: 'Post-Event Analysis: True vs Predicted Lap Times' }
        },
        scales: {
          x: { title: { display: true, text: 'Lap #' } },
          y: { title: { display: true, text: 'Lap Time (s)' } }
        }
      }
    });
  }

  private destroyAllCharts() {
    Object.values(this.charts).forEach(chart => {
      if (chart) {
        chart.destroy();
      }
    });
    this.charts = {}; // Clear the charts object
  }

  ngOnDestroy() {
    Object.values(this.charts).forEach(chart => chart.destroy());
  }

  computeTrendline(x: number[], y: number[]) {
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((a, b, i) => a + b * y[i], 0);
    const sumX2 = x.reduce((a, b) => a + b * b, 0);
    const m = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX ** 2);
    const b = (sumY - m * sumX) / n;
    return { m, b };
  }

  getPostEventInsight(mae: number, rmse: number, r2: number): string {
    if (rmse < 0.5 && r2 > 0.95)
      return '🏎️ Excellent predictive accuracy — model closely matches actual lap evolution.';
    if (rmse < 1.0 && r2 > 0.9)
      return '⚙️ Strong model performance — minor deviations, but pacing trend is well captured.';
    if (rmse < 1.5)
      return '📉 Moderate fit — overall pattern correct, but larger lap-to-lap variation.';
    return '⚠️ Weak agreement — model drifted from actual race behavior. Consider retraining.';
  }

  openNewSession() {
    this.activeMode = 'newSession';
    this.newSessionOpen = true;
    this.newSessionResult = null;
    // set defaults
    this.newSessionForm = {
      track_name: this.selectedTrack || 'Barber',
      vehicle_id: '',
      total_laps_to_predict: 10,
      previous_laps: []
    };
  }

  addPrevLapRow() {
    this.newSessionForm.previous_laps!.push({ lap: (this.newSessionForm.previous_laps!.length + 1), lap_time_seconds: 100 });
  }

  removePrevLapRow(idx: number) {
    this.newSessionForm.previous_laps!.splice(idx, 1);
  }

  async predictNewSession() {
    // basic validation
    if (!this.newSessionForm.track_name || !this.newSessionForm.vehicle_id || !this.newSessionForm.total_laps_to_predict) {
      this.errorMessage = 'Fill track, vehicle id and total laps to predict.';
      return;
    }

    // convert empty lap_time_seconds to 0 -> backend will handle
    this.newSessionForm.previous_laps = this.newSessionForm.previous_laps!.map(p => ({
      lap: Number(p.lap),
      lap_time_seconds: p.lap_time_seconds ? Number(p.lap_time_seconds) : 0,
      laps_on_tires: p.laps_on_tires ? Number(p.laps_on_tires) : undefined,
      fuel_load_proxy: p.fuel_load_proxy ? Number(p.fuel_load_proxy) : undefined,
      session_air_temp: p.session_air_temp ? Number(p.session_air_temp) : undefined,
      session_track_temp: p.session_track_temp ? Number(p.session_track_temp) : undefined,
    }));

    this.isLoading = true;
    try {
      const payload: PredictNewRaceRequest = {
        track_name: this.newSessionForm.track_name,
        vehicle_id: this.newSessionForm.vehicle_id,
        total_laps_to_predict: Number(this.newSessionForm.total_laps_to_predict),
        previous_laps: this.newSessionForm.previous_laps
      };

      const res = await firstValueFrom(this.predictionService.predictNewRaceSession(payload));
      this.newSessionResult = res;
      // optional: show results in the main area or a modal
      console.log('newSessionResult', res);
      this.cd.detectChanges();
    } catch (err) {
      console.error('predictNewSession failed', err);
      this.errorMessage = 'Failed to predict new race session';
    } finally {
      this.isLoading = false;
    }
  }

}
