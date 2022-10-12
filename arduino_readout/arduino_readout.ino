const int M = 5; // For a squared array of 5x5
int analog_pins[M] = { A0, A1, A2, A3, A4 };

const int N = M * M; // Number of taxels
uint16_t values[N];
float v[N];
float v_prev[N];
float u[N];
float u_prev[N];

unsigned long t;
float Fs = 1000; // Hz
float dt = (1 / Fs) * 1E6;  // Sampling period in microseconds

// Izhikevich parameters
const float A = 0.02;
const float B = 0.25;
const float V_theta = -55;
const float V_res = -65;
const float D = 0.05;
const float K = 1;
const float MAX_POTENTIAL = 30.0;

// Input current
float I = 0.0;

void setup() {
  // Setup digital write pins
  for (int p = 0; p < M; ++p) {
    pinMode(p, OUTPUT);
  }

  // starting voltages
  for (int i = 0; i < N; ++i) {
    v[i] = V_res;
    v_prev[i] = V_res;
    u[i] = B * V_res;
    u_prev[i] = B * V_res;
  }

  Serial.begin(115200);
}

// Analogue-to-Spikes Converter
void asc() {
  // Izhikevich model
  for (int i = 0; i < N; ++i) {
    I = (float)values[i];
    v[i] = v_prev[i] + (0.04 * pow(v_prev[i], 2) + 5.0 * v_prev[i] + 140.0 - u_prev[i] + K * I);
    u[i] = u_prev[i] + (A * (B * v[i] - u_prev[i]));

    if (v[i] > MAX_POTENTIAL) {
      v[i] = V_theta;
      u[i] = u[i] + D;
    }
    v_prev[i] = v[i];
    u_prev[i] = u[i];
  }
}

inline void readout(int idx) {
  for (int i = 0; i < M; ++i) {
    if (i != idx) {
      digitalWrite(i, LOW);
    } else {
      digitalWrite(i, HIGH);
    }
  }
  for (int j = 0; j < M; ++j) {
    values[j + idx * M] = analogRead(analog_pins[j]);
  }
}

void loop() {
  // Read pressure values
  for (int r = 0; r < M; ++r) {
    readout(r);
  }

  asc();

  // Print analogue values
  for (int i = 0; i < N; ++i) {
    Serial.print(values[i]);
    Serial.print(' ');
  }
  Serial.write('\n');

  while (micros() - t < dt) {
    // Match the sampling period
  }
}
