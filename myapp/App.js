import React, { useEffect, useState } from "react";
import { StyleSheet, Text, View } from "react-native";
import { useLocation } from "./use_location";
import axios from "axios";
import Constants from "expo-constants";
import { getBusRoute } from "./get_bus_route";
const { myApiUrl } = Constants.expoConfig.extra;

export default function App() {
  const [message, setMessage] = useState("");
  const { location, errorMsg, loading } = useLocation();

  useEffect(() => {
    axios
      .get(`http://${myApiUrl}:8000/api/hello/`)
      .then((response) => {
        setMessage(response.data.message);
      })
      .catch((error) => {
        console.log(error);
      });
  }, []);

  if (loading) {
    return <Text>Getting location...</Text>;
  }

  if (errorMsg) {
    return <Text>{errorMsg}</Text>;
  }


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
