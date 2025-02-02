import {
  Client,
  TravelMode,
  TransitMode,
} from "@googlemaps/google-maps-services-js";
import axios from "axios";
import Constants from "expo-constants";
const { googleMapsApiKey } = Constants.expoConfig.extra;

// Creating the Google Maps client
const client = new Client({});

const formatTime = (ms) => {
  console.log(ms)
  // We don't care about seconds here, may be incorrect
  const minutes = Math.floor((ms / (1000 * 60)) % 60);
  const hours = Math.floor((ms / (1000 * 60 * 60)) % 24);

  let timeString = "";

  if (hours > 0) {
    timeString += `${hours} hours`;
  }

  if (minutes > 0) {
    timeString += `${minutes} minutes`;
  }

  return timeString;
};

export async function getBusRoute(longitude, latitude, destination) {
  const url = `https://maps.googleapis.com/maps/api/directions/json?origin=${latitude},${longitude}&destination=${destination}&mode=transit&transit_mode=bus&transit_routing_preference=less_walking&key=${googleMapsApiKey}`

  try {
    const response = await axios.get(url);
    const data = response.data;
    console.log("Geological data:", data)
    if (data.status == "OK") {
      
      const route = response.data.routes[0];
      const directions = route.legs[0].steps[0];
      if (directions.travel_mode != "TRANSIT") {
        throw new Error("First travel mode isn't a bus route");
      }
  
      const now = Date.now();
      // Need to convert into milliseconds
      const departTime = directions.transit_details.departure_time.value * 1000;
      const timeUntilBus = departTime - now;
  
      return {
        status: "OK",
        busNumber: directions.transit_details.line.short_name,
        stopToGetOffAt: directions.transit_details.arrival_stop.name,
        busReachingIn: formatTime(timeUntilBus),
      };
    } 

    throw new Error("Status is not OK");
  } catch (error) {
    console.log('Error getting bus route:', error);
    return {
      status: "ERROR",
    }
  }
}
