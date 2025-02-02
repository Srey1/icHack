import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  Button,
  StyleSheet,
  Alert,
  ActivityIndicator,
  Dimensions,
  TouchableOpacity 
} from "react-native";
import { Audio } from "expo-av";
import * as FileSystem from "expo-file-system";
import { useLocation } from "./use_location";
import axios from "axios";
import { MaterialIcons } from '@expo/vector-icons';
import * as Speech from 'expo-speech';
import Constants from "expo-constants";
import { getBusRoute } from "./get_bus_route";
const { myApiUrl } = Constants.expoConfig.extra;
import { createPrompt } from "./user_prompt";

const {googleApiKey} = Constants.expoConfig.extra;

const { width, height } = Dimensions.get('window');

// Replace with your Google Cloud API Key
const GOOGLE_API_KEY = googleApiKey;

export default function App() {
  const { location, errorMsg, loading } = useLocation();
  const [recording, setRecording] = useState(null);
  const [recordedUri, setRecordedUri] = useState(null);
  const [transcribedText, setTranscribedText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [permissionResponse, requestPermission] = Audio.usePermissions();
  const [sound, setSound] = useState(null);

  const [busNumber, setBusNumber] = useState(null);
  console.log(GOOGLE_API_KEY);
  useEffect(() => {
    (async () => {
      if (!permissionResponse || permissionResponse.status !== "granted") {
        console.log("Requesting microphone permissions...");
        await requestPermission();
      }
    })();
  }, []);


  useEffect(() => {
    axios
      .post(`http://${myApiUrl}:8000/api/bus/`, busNumber
      )
      .then((response) => {
        console.log(response.data.message);
      })
      .catch((error) => {
        console.log(error);
      });
  }, [busNumber]);

  const speak = (text) => {
    Speech.speak(text);
  };

  

  // 🎤 Start Recording
  const startRecording = async () => {
    try {
      if (!permissionResponse || permissionResponse.status !== "granted") {
        Alert.alert("Permission Required", "Microphone access is needed.");
        return;
      }

      console.log("Setting audio mode...");
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      console.log("Starting recording...");
      const { recording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      );

      setRecording(recording);
      console.log("Recording started.");
    } catch (err) {
      console.error("Failed to start recording:", err);
    }
  };

  const stopRecording = async () => {
    try {
      console.log("Stopping recording...");
      if (!recording) return;
  
      await recording.stopAndUnloadAsync();
      const uri = recording.getURI();
      console.log("Recording saved at:", uri);
  
      setRecording(null);
      setRecordedUri(uri);
  
      // Ensure the state updates before transcribing
      setTimeout(() => {
        transcribeAudio(uri);
      }, 500); // Small delay to allow state update
    } catch (error) {
      console.error("Error stopping recording:", error);
    }
  };
  


  const transcribeAudio = async (uri) => {
    try {
      if (!uri) {
        Alert.alert("No Recording", "Please record something first.");
        return;
      }
  
      console.log("Reading file as Base64...");
      setIsLoading(true);
      const base64Audio = await FileSystem.readAsStringAsync(uri, {
        encoding: FileSystem.EncodingType.Base64,
      });
  
      console.log("Sending request to Google Speech-to-Text API...");
      const requestBody = {
        config: {
          encoding: "WEBM_OPUS", // Adjust as needed
          sampleRateHertz: 48000,
          languageCode: "en-US",
        },
        audio: {
          content: base64Audio,
        },
      };
  
      const response = await axios.post(
        `https://speech.googleapis.com/v1/speech:recognize?key=${GOOGLE_API_KEY}`,
        requestBody
      );
  
      console.log("Full Google API Response:", JSON.stringify(response.data, null, 2));
  
      if (!response.data.results || response.data.results.length === 0) {
        setTranscribedText("No speech detected. Try speaking louder or recording for longer.");
        return;
      }
  
      // Extract transcript text
      const transcription = response.data.results
        .map((result) => result.alternatives?.[0]?.transcript || "")
        .join(" ");
  
      // Claude API call
      const promptResponse = await createPrompt(transcription);
      console.log("Prompt Response:", promptResponse.response);

      if (promptResponse.type == "location") {
        const locationCoords = loading.coords;
        const busResponse = await getBusRoute(
          locationCoords.longitude, 
          locationCoords.latitude, 
          promptResponse.response
        );
        
        setTranscribedText(busResponse.busNumber || "Couldn't find bus route");
      } else {
        setTranscribedText(promptResponse.response || "No transcription found.");
      }

      if (busNumber) {
        axios.post(`http://${myApiUrl}:8000/api/bus/`, String.toString(busNumber)).then((response) => {
          console.log(response.data.message);
        }).catch((error) => {
          console.log(error);
        });
      }

    } catch (error) {
      console.error("Error transcribing audio:", error.response?.data || error);
      Alert.alert("Error", "Could not transcribe audio.");
    } finally {
      setIsLoading(false);
    }
  };
  

  return (
    <View style={[styles.container, recording && styles.recordingBackground]}>
      
      {/* 📜 Large Display Transcription */}
      {transcribedText ? (
        <Text style={styles.largeTranscription}>{transcribedText}</Text>
      ) : (
        <Text style={styles.largeTranscription}>---</Text> // Placeholder for visibility
      )}
  
      {/* 🎤 Start/Stop Recording */}
      <TouchableOpacity
        accessible={true}
        accessibilityLabel={recording ? 'Stop recording' : 'Start recording'}
        accessibilityRole="button"
        accessibilityHint="Double tap to start or stop audio recording"
        onPress={recording ? stopRecording : startRecording}
        style={[styles.button, recording && styles.buttonActive]}
        activeOpacity={0.7}
      >
        <MaterialIcons 
          name="mic"
          size={height * 0.2}
          color={recording ? '#CC0000' : '#FFFFFF'} // Red icon when recording, white when not
          accessible={false}
        />
      </TouchableOpacity>
  
      {/* 🎤 Recording Status */}
      <Text style={[styles.statusText, recording && styles.statusTextRecording]}>
        {recording ? 'RECORDING' : 'Tap microphone to start'}
      </Text>
  
      {/* ⏳ Loading Indicator */}
      {isLoading && <ActivityIndicator size="large" color="blue" style={{ marginTop: 10 }} />}
    </View>
  );
  
  
  
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
  },
  recordingBackground: {
    backgroundColor: '#CC0000',
  },
  button: {
    backgroundColor: '#CC0000', // Red button when not recording
    width: width * 0.8,
    height: width * 0.8,
    borderRadius: width * 0.4,
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 6,
    borderWidth: 4,
    borderColor: '#FFFFFF', // White border when not recording
  },
  buttonActive: {
    backgroundColor: '#FFFFFF', // White button when recording
    borderColor: '#CC0000', // Red border when recording
  },
  statusText: {
    position: 'absolute',
    bottom: height * 0.1,
    fontSize: 28,
    fontWeight: 'bold',
    color: '#CC0000',
    textAlign: 'center',
  },
  statusTextRecording: {
    color: '#FFFFFF',
    fontSize: 34,
    textShadowColor: 'rgba(0, 0, 0, 0.2)',
    textShadowOffset: { width: 1, height: 1 },
    textShadowRadius: 2,
  },
  largeTranscription: {
    fontSize: 50, // Large size for visibility
    fontWeight: 'bold',
    color: '#000000', // High contrast
    textAlign: 'center',
    marginBottom: 20, // Space before microphone button
  },
});
