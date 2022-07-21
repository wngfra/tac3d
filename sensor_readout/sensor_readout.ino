int pin1 = A1;
int pin2 = A2;
int pin3 = A3;

const int N = 9; //number of taxiles
float values[N];

unsigned long t;
float sampling_rate = 1000;
float samp_period = (1/sampling_rate)*1E6; //samp_period in microseconds

// izhikevich parameters
const float a = 0.02;
const float b = 0.25;
const float c = -55;
const float d = 0.05;
const float k = 1;
const float max_potential = 30.0;

// starting voltages
float v[9] = { -65, -65, -65, -65, -65, -65, -65, -65, -65};
float v_prev[9] = { -65, -65, -65, -65, -65, -65, -65, -65, -65};
float u[9] = {b * -65, b * -65, b * -65 , b * -65, b * -65, b * -65, b * -65, b * -65, b * -65};
float u_prev[9] = {b * -65, b * -65, b * -65, b * -65, b * -65, b * -65, b * -65, b * -65, b * -65};
// input current
float I = 0.0;

void setup() {
  // setup digital write pins
  pinMode(1, OUTPUT);
  pinMode(2, OUTPUT);
  pinMode(3, OUTPUT);
}

void analog2spike() {
  // izhikevich model
  for (int i = 0; i < 9; i++) {
    I = values[i];
    v[i] = v_prev[i] + (0.04 * pow(v_prev[i], 2) + 5.0 * v_prev[i] + 140.0 - u_prev[i] + k * I);
    u[i] = u_prev[i] + (a * (b * v[i] - u_prev[i]));

    if (v[i] > max_potential) {
      v[i] = c;
      u[i] = u[i] + d;
      // maybe set v prev to the max_potential ?
    }
    v_prev[i] = v[i];
    u_prev[i] = u[i];
  }
}

void loop() {
  // obtain pressure values
  digitalWrite(1, HIGH);  
  digitalWrite(2, LOW);
  digitalWrite(3, LOW);
  values[0] = analogRead(pin1);
  values[1] = analogRead(pin2);
  values[2] = analogRead(pin3);

  digitalWrite(1, LOW);
  digitalWrite(2, HIGH);
  digitalWrite(3, LOW);
  values[3] = analogRead(pin1);
  values[4] = analogRead(pin2);
  values[5] = analogRead(pin3);

  digitalWrite(1, LOW);
  digitalWrite(2, LOW);
  digitalWrite(3, HIGH);
  values[6] = analogRead(pin1);
  values[7] = analogRead(pin2);
  values[8] = analogRead(pin3);

  analog2spike();
  
  // raw analog values
  for (int i = 0; i < 9; i++) {
    Serial.print(values[i]);
    Serial.print(' ');
    delayMicroseconds(10);
  }
  Serial.println();  

  while (micros()-t < samp_period){
    // wait until its time to make sure each loop happens in a sampling period
  }
}
