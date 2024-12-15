#define RXp2 16
#define TXp2 17

#define MIN_CURRENT 0.0    // Minimum current limit
#define MAX_CURRENT 19.0   // Maximum current limit



#include <WiFi.h>
#include <FirebaseESP32.h>

#define WIFI_SSID "Prakash"
#define WIFI_PASSWORD "1234567890"

// Firebase credentials
#define FIREBASE_HOST "https://museum-d67ee-default-rtdb.firebaseio.com/"  // Database URL
#define FIREBASE_AUTH "HoK1IFj2eIqYsaHFVxZ54w3FH0RITWqhFe6EI9dF"           // Database secret key

FirebaseData firebaseData;
FirebaseConfig firebaseConfig;
FirebaseAuth firebaseAuth;



void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");

  firebaseConfig.host = FIREBASE_HOST;
  firebaseConfig.signer.tokens.legacy_token = FIREBASE_AUTH;

  // Initialize Firebase
  Firebase.begin(&firebaseConfig, &firebaseAuth);
  Firebase.reconnectWiFi(true);

  Serial2.begin(9600, SERIAL_8N1, RXp2, TXp2);
  Serial.println("Setup complete. Waiting for data...");
}

void loop() {
  if (Serial2.available() > 0) {
    String receivedData = Serial2.readStringUntil('\n');  // Read the incoming data until newline
    float current = receivedData.toFloat();               // Convert the received data to float
    float current1 = current - 0.01;
    Serial.print("Received current: ");
    Serial.print(current1);
    Serial.println(" A");  // Print the received data to the Serial Monitor

    String Status=" ";
    // Check if the current is out of the specified range and print an alert message if it is
    if (current1 < MIN_CURRENT || current1 > MAX_CURRENT) {
      Serial.println("ALERT: Current out of range!");
      Status="Defective";
      
    } else {
      Serial.println("Current is within the safe range.");
      Status="Normal";
    }
    
    FirebaseJson json;

    json.set("Type",type);
    json.set("Current",current1);
    json.set("Status",Status);


    if (Firebase.pushJSON(firebaseData, "/machine_data", json)) {
    Serial.println("Data sent successfully!");
    Serial.println(firebaseData.pushName()); // Log unique key for the data
    } else {
    Serial.println("Failed to send data.");
    Serial.println(firebaseData.errorReason());
  }


  } 

  // Yield to prevent watchdog timer reset (small delay to yield control back to the system)
  delay(10);
}