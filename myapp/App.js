import React, { useEffect, useState } from "react";
import { StyleSheet, Text, View } from "react-native";
import { createPrompt } from "./user_prompt";
import axios from "axios";
import Constants from "expo-constants";
const {myAPIUrl} = Constants.expoConfig.extra;

export default function App() {
  const [message, setMessage] = useState("");

  useEffect(() => {
    axios
      .get(`http://${myAPIUrl}:8000/api/hello/`)
      .then((response) => {
        setMessage(response.data.message);
      })
      .catch((error) => {
        console.log(error.response);
      });
  }, []);

  return (
    <View style={styles.container}>
      <Text>{message}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
    alignItems: "center",
    justifyContent: "center",
  },
});
