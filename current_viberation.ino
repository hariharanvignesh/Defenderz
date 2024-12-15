#include <WiFi.h>
#include <FirebaseESP32.h>

// WiFi credentials
#define WIFI_SSID "Prakash"                 // Your WiFi SSID
#define WIFI_PASSWORD "1234567890"         // Your WiFi Password

// Firebase credentials
#define FIREBASE_HOST "https://museum-d67ee-default-rtdb.firebaseio.com/"  // Firebase URL
#define FIREBASE_AUTH "HoK1IFj2eIqYsaHFVxZ54w3FH0RITWqhFe6EI9dF"           // Firebase secret key

// Define pins for components
#define MAIN_SWITCH 2       // Rocker switch pin
#define SECONDARY_SWITCH 4  // Push button pin
#define MOTOR_IN1 12        // Motor direction pin 1
#define MOTOR_IN2 14        // Motor direction pin 2
#define LED_PIN 13          // LED pin
#define VIBRATION_SENSOR 15 // Vibration sensor pin

// Firebase objects
FirebaseData firebaseData;
FirebaseConfig firebaseConfig;
FirebaseAuth firebaseAuth;

// Variables
float currentValue = 0.0;    // Placeholder for current value
String machine_id = "";      // Random machine ID
String floorLevel = "1st Floor";
int type = 2;                // Type: 1 (Motion Only), 2 (Current + Vibration)

void setup() {
  Serial.begin(9600);

  // Pin modes
  pinMode(MAIN_SWITCH, INPUT_PULLUP);
  pinMode(SECONDARY_SWITCH, INPUT_PULLUP);
  pinMode(VIBRATION_SENSOR, INPUT);
  pinMode(MOTOR_IN1, OUTPUT);
  pinMode(MOTOR_IN2, OUTPUT);
  pinMode(LED_PIN, OUTPUT);

  // Ensure motor and LED are off initially
  digitalWrite(MOTOR_IN1, LOW);
  digitalWrite(MOTOR_IN2, LOW);
  digitalWrite(LED_PIN, LOW);

  // Random seed initialization
  randomSeed(analogRead(0));
  machine_id = "Machine " + String(random(1, 11)); // Random Machine ID between 1 and 10

  // Connect to WiFi
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");

  // Firebase setup
  firebaseConfig.host = FIREBASE_HOST;
  firebaseConfig.signer.tokens.legacy_token = FIREBASE_AUTH;
  Firebase.begin(&firebaseConfig, &firebaseAuth);
  Firebase.reconnectWiFi(true);

  Serial.println("Setup complete. Waiting for input...");
}

void loop() {
  // Read the state of the switches and vibration sensor
  bool mainSwitchState = digitalRead(MAIN_SWITCH) == LOW;           // Rocker switch ON
  bool secondarySwitchState = digitalRead(SECONDARY_SWITCH) == LOW; // Push button pressed
  bool vibrationState = digitalRead(VIBRATION_SENSOR) == HIGH;      // Vibration detected

  // Default Type and Current values
  type = 0;
  currentValue = 0.0;

  if (mainSwitchState) {  // Main switch is ON
    Serial.println("Main switch is ON.");

    if (secondarySwitchState) {
      // Secondary switch is ON, run motor and LED
      Serial.println("Secondary switch ON. Motor and LED activated.");
      digitalWrite(MOTOR_IN1, HIGH);
      digitalWrite(MOTOR_IN2, LOW); // Motor direction
      digitalWrite(LED_PIN, HIGH);

      // Generate random positive current value (0.01 to 0.99)
      currentValue = random(1, 100) / 100.0;

      // Set type based on vibration sensor
      if (vibrationState) {
        Serial.println("Vibration detected. Setting Type to 2 (Current + Vibration).");
        type = 2;  // Vibration and current detected
      } else {
        Serial.println("No vibration detected. Setting Type to  (Motion Only).");
        type = 2;  // Only current detected
      }

      // Send data to Firebase
      sendFirebaseData(currentValue, type, vibrationState);
    } else {
      // Secondary switch OFF: Motor and LED inactive
      Serial.println("Secondary switch OFF. Motor and LED inactive.");
      digitalWrite(MOTOR_IN1, LOW);
      digitalWrite(MOTOR_IN2, LOW);
      digitalWrite(LED_PIN, LOW);

      // Current remains 0.00
      if (vibrationState) {
        Serial.println("Vibration detected, but no current flow. Setting Type to 2.");
        type = 2;
      } else {
        Serial.println("No vibration detected and no current flow. Setting Type to 0.");
        type = 0;
      }
      sendFirebaseData(0.0, type, vibrationState);
    }
  } else {
    // Main switch is OFF: Turn off motor and LED
    Serial.println("Main switch is OFF. Motor and LED deactivated.");
    digitalWrite(MOTOR_IN1, LOW);
    digitalWrite(MOTOR_IN2, LOW);
    digitalWrite(LED_PIN, LOW);

    // Current remains 0.00
    if (vibrationState) {
      Serial.println("Vibration detected while main switch is OFF. Setting Type to 2.");
      type = 2;
    } else {
      Serial.println("No vibration detected and main switch is OFF. Setting Type to 0.");
      type = 0;
    }
    sendFirebaseData(0.0, type, vibrationState);
  }

  delay(1000); // Delay for stability and debounce handling
}

void sendFirebaseData(float current, int type, bool vibration) {
  // Determine working/not working status
  String workingStatus = (current > 0 && vibration) ? "Working" : "Defective";

  // Create JSON structure
  FirebaseJson json;
  json.set("Machine_id", machine_id);
  json.set("Floor", floorLevel);
  json.set("Type", type);
  json.set("Current", current);
  json.set("Vibration", vibration ? 1 : 0);
  json.set("Status", workingStatus);

  // Send data to Firebase
  if (Firebase.pushJSON(firebaseData, "/machine_data", json)) {
    Serial.println("Data sent successfully!");
    Serial.println(firebaseData.pushName());
  } else {
    Serial.println("Failed to send data.");
    Serial.println(firebaseData.errorReason());
  }
}
