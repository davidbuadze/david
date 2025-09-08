
import { getMessaging, getToken } from "firebase/messaging";

// Функция для отправки токена на бэкенд
const sendTokenToBackend = async (token, auth) => {
  if (!auth || !auth.currentUser) {
    console.error("User is not authenticated. Cannot send FCM token.");
    return;
  }

  console.log("Sending FCM token to backend...");
  try {
    const idToken = await auth.currentUser.getIdToken(true);
    const backendUrl = `${process.env.REACT_APP_BACKEND_BASE_URL}/users/save-fcm-token`;

    const response = await fetch(backendUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${idToken}`,
      },
      body: JSON.stringify({ fcmToken: token }),
    });

    if (response.ok) {
      console.log("FCM token successfully sent to backend.");
      alert("Токен для уведомлений успешно сохранен на сервере.");
    } else {
      const errorData = await response.json();
      console.error("Failed to send FCM token to backend:", response.status, errorData);
      alert(`Ошибка при сохранении токена на сервере: ${errorData.message || response.statusText}`);
    }
  } catch (error) {
    console.error("Error sending FCM token to backend:", error);
    alert(`Сетевая ошибка при отправке токена: ${error.message}`);
  }
};

// Функция для запроса разрешения на уведомления и получения токена
export const requestNotificationPermission = (app, auth) => { // Добавлен auth
  console.log("Requesting notification permission...");
  const messaging = getMessaging(app);

  const vapidKey = process.env.REACT_APP_FIREBASE_VAPID_KEY;

  if (!vapidKey) {
    console.error("VAPID key is missing. Please set REACT_APP_FIREBASE_VAPID_KEY in your .env file.");
    alert("Configuration error: VAPID key for notifications is not set.");
    return;
  }

  getToken(messaging, { vapidKey: vapidKey }).then((currentToken) => {
    if (currentToken) {
      console.log("====================================");
      console.log("FCM Registration Token (Device Token):");
      console.log(currentToken);
      console.log("====================================");
      
      // ОТПРАВЛЯЕМ ТОКЕН НА БЭКЕНД
      sendTokenToBackend(currentToken, auth);

    } else {
      console.log("No registration token available. Request permission to generate one.");
      alert("Не удалось получить токен. Убедитесь, что вы разрешили отправку уведомлений для этого сайта.");
    }
  }).catch((err) => {
    console.error("An error occurred while retrieving token. ", err);
    alert(`Произошла ошибка при получении токена: ${err.message}`);
  });
};
