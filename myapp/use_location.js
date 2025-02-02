import { useEffect, useState } from "react";
import * as Location from 'expo-location';

export const useLocation = () => {
    const [location, setLocation] = useState(null);
    const [errorMsg, setErrorMsg] = useState(null);
    const [loading, setLoading] = useState(true);
  
    useEffect(() => {
      (async () => {
        try {
          // Request permission to access location
          const { status } = await Location.requestForegroundPermissionsAsync();
          
          if (status !== 'granted') {
            setErrorMsg('Permission to access location was denied');
            setLoading(false);
            return;
          }
  
          // Get current position
          const currentLocation = await Location.getCurrentPositionAsync({
            accuracy: Location.Accuracy.High,
          });
  
          setLocation(currentLocation);
          setLoading(false);
        } catch (error) {
          setErrorMsg('Error getting location: ' + error.message);
          setLoading(false);
        }
      })();
    }, []);
  
    return { location, errorMsg, loading };
  };