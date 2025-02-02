export default ({ config }) => {
    return {
      ...config,
      extra: {
        myApiUrl: process.env.BACKEND_IP || '',
        anthropicApiKey: process.env.ANTHROPIC_API_KEY || '',
        googleMapsApiKey: process.env.GOOGLE_MAPS_API_KEY || '',
      }
    };
  };