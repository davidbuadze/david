import React, { useState, useEffect, createContext, useContext } from 'react';
import { initializeApp } from 'firebase/app';
import {
  getAuth,
  GoogleAuthProvider,
  signInWithPopup,
  getRedirectResult,
  signOut,
  onAuthStateChanged,
  signInAnonymously,
  signInWithCustomToken,
} from 'firebase/auth';
import { getFirestore, doc, setDoc, getDoc, collection, addDoc } from 'firebase/firestore';
import { getAnalytics } from 'firebase/analytics';
import { requestNotificationPermission } from './firebaseMessaging';

const appId = typeof __app_id !== 'undefined' ? __app_id : 'default-app-id';
const firebaseConfigFromCanvas = typeof __firebase_config !== 'undefined' ? JSON.parse(__firebase_config) : null;
const initialAuthTokenFromCanvas = typeof __initial_auth_token !== 'undefined' ? initialAuthTokenFromCanvas : null;

// Теперь конфигурация Firebase берется из переменных окружения
const YOUR_FIREBASE_CONFIG = {
  apiKey: process.env.REACT_APP_FIREBASE_API_KEY,
  authDomain: process.env.REACT_APP_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.REACT_APP_FIREBASE_PROJECT_ID,
  storageBucket: process.env.REACT_APP_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.REACT_APP_FIREBASE_MESSAGING_SENDER_ID,
  appId: process.env.REACT_APP_FIREBASE_APP_ID,
  measurementId: process.env.REACT_APP_FIREBASE_MEASUREMENT_ID
};

// URL бэкенда также берется из переменной окружения
const BACKEND_BASE_URL = process.env.REACT_APP_BACKEND_URL;
console.log("BACKEND_BASE_URL:", BACKEND_BASE_URL);

const AuthContext = createContext(null);
const useAuth = () => useContext(AuthContext);

const AuthProvider = ({ children }) => {
  const [currentUser, setCurrentUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [auth, setAuth] = useState(null);
  const [db, setDb] = useState(null);
  const [app, setApp] = useState(null);
  const [authError, setAuthError] = useState(null);
  const [authInitialized, setAuthInitialized] = useState(false);

  useEffect(() => {
    const initFirebaseAndAuth = async () => {
      try {
        let firebaseConfigToUse = null;
        let isCanvasEnvironment = false;

        if (firebaseConfigFromCanvas && Object.keys(firebaseConfigFromCanvas).length > 0) {
          firebaseConfigToUse = firebaseConfigFromCanvas;
          isCanvasEnvironment = true;
          console.log("AuthProvider: Используется конфигурация Firebase из среды Canvas.");
        } else if (YOUR_FIREBASE_CONFIG.apiKey) { // Проверяем, что API Key определен
          firebaseConfigToUse = YOUR_FIREBASE_CONFIG;
          console.log("AuthProvider: Используется конфигурация Firebase из переменных окружения.");
        } else {
          setAuthError("AuthProvider: Конфигурация Firebase не найдена или не заполнена.");
          setLoading(false);
          return;
        }

        const appInstance = initializeApp(firebaseConfigToUse);
        setApp(appInstance);
        const authInstance = getAuth(appInstance);
        const firestoreInstance = getFirestore(appInstance);
        getAnalytics(appInstance);

        setAuth(authInstance);
        setDb(firestoreInstance);

        console.log("AuthProvider: Инициализация Firebase и установка слушателей.");

        const unsubscribe = onAuthStateChanged(authInstance, (user) => {
          console.log("onAuthStateChanged: Обнаружено изменение состояния пользователя. Пользователь:", user ? user.uid : "Не автентифицирован");
          setCurrentUser(user);
          setAuthInitialized(true);
          setLoading(false);
        });

        console.log("AuthProvider: Проверка результата перенаправления Google Sign-In (для совместимости).");
        getRedirectResult(authInstance)
          .then((result) => {
            if (result) {
              console.log("getRedirectResult: Перенаправление Google Sign-In успешно. Пользователь:", result.user.uid);
            } else {
              console.log("getRedirectResult: Нет активного результата перенаправления.");
            }
          })
          .catch(error => {
            console.error("getRedirectResult: Ошибка при получении результата перенаправления Google Sign-In:", error);
            setAuthError(`Ошибка входа через Google: ${error.message}`);
          });

        if (isCanvasEnvironment && initialAuthTokenFromCanvas) {
          console.log("AuthProvider: Попытка входа с использованием пользовательского токена Canvas.");
          await signInWithCustomToken(authInstance, initialAuthTokenFromCanvas);
        } else if (!isCanvasEnvironment) {
          console.log("AuthProvider: Приложение развернуто. Пользователь должен войти вручную.");
        } else {
          console.log("AuthProvider: Вход выполнен анонимно (среда Canvas, без пользовательского токена).");
          await signInAnonymously(authInstance);
        }

        return () => unsubscribe();
      } catch (err) {
        console.error("AuthProvider: Ошибка инициализации Firebase или входа:", err);
        setAuthError(`Ошибка инициализации приложения: ${err.message}`);
        setLoading(false);
        setAuthInitialized(true);
      }
    };

    initFirebaseAndAuth();
  }, []);

  const value = {
    currentUser,
    auth,
    db,
    app,
    authError,
  };

  return (
    <AuthContext.Provider value={value}>
      {loading || !authInitialized ? (
        <div className="min-h-screen flex items-center justify-center bg-gray-100 dark:bg-blue-950 text-gray-800 dark:text-gray-100">
          Загрузка аутентификации...
        </div>
      ) : (
        children
      )}
    </AuthContext.Provider>
  );
};

const ThemeContext = createContext(null);
const useTheme = () => useContext(ThemeContext);

const ThemeProvider = ({ children }) => {
  const [isDarkMode, setIsDarkMode] = useState(true);

  const toggleTheme = () => {
    setIsDarkMode(prevMode => !prevMode);
  };

  const theme = isDarkMode ? 'dark' : 'light';

  const value = {
    theme,
    toggleTheme,
    isDarkMode
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
};

const Auth = () => {
  const { auth, authError } = useAuth();
  const { isDarkMode } = useTheme();
  const [error, setError] = useState('');

  const handleGoogleSignIn = async () => {
    if (!auth) {
      setError("Firebase Auth не инициализирован. Пожалуйста, подождите или проверьте конфигурацию.");
      return;
    }
    const provider = new GoogleAuthProvider();
    try {
      console.log("Auth: Запуск входа через Google (всплывающее окно).");
      const result = await signInWithPopup(auth, provider);
      console.log("Auth: Вход через всплывающее окно успешен. Пользователь:", result.user.uid);
      setError('');
    } catch (err) {
      setError(err.message);
      console.error("Auth: Ошибка входа через Google (всплывающее окно):", err);
    }
  };

  const bgColor = isDarkMode ? 'bg-blue-950' : 'bg-gray-100';
  const cardBg = isDarkMode ? 'bg-blue-900' : 'bg-white';
  const textColor = isDarkMode ? 'text-gray-100' : 'text-gray-900';
  const buttonBg = isDarkMode ? 'bg-blue-800' : 'bg-white';
  const buttonText = isDarkMode ? 'text-gray-100' : 'text-gray-700';
  const buttonBorder = isDarkMode ? 'border-blue-700' : 'border-gray-300';
  const buttonHoverBg = isDarkMode ? 'hover:bg-blue-700' : 'hover:bg-gray-50';

  return (
    <div className={`min-h-screen flex items-center justify-center p-4 ${bgColor}`}>
      <div className={`p-8 rounded-lg shadow-lg w-full max-w-md ${cardBg}`}>
        <h2 className={`text-2xl font-bold text-center mb-6 ${textColor}`}>
          Войти в приложение
        </h2>

        {(error || authError) && (
          <p className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4" role="alert">
              {error || authError}
          </p>
        )}

        <button
          onClick={handleGoogleSignIn}
          className={`w-full flex items-center justify-center px-4 py-2 rounded-md shadow-sm text-sm font-medium focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 mb-4 ${buttonBg} ${buttonText} ${buttonBorder} ${buttonHoverBg}`}
        >
          <svg className={`w-5 h-5 mr-2 ${buttonText}`} viewBox="0 0 24 24" fill="currentColor">
            <path d="M12.24 10.27c-.23 0-.46-.02-.68-.06a4.57 4.57 0 0 0-.67-.17c-.24-.05-.48-.09-.72-.11a4.57 4.57 0 0 0-.7-.09c-.25-.03-.5-.04-.75-.04c-2.42 0-4.4 1.98-4.4 4.4s1.98 4.4 4.4 4.4c2.14 0 3.93-1.54 4.3-3.56h-3.92v-2.02h6.76c.07.41.11.83.11 1.25c0 4.1-3.34 7.44-7.44 7.44c-4.1 0-7.44-3.34-7.44-7.44s3.34-7.44 7.44-7.44c2.14 0 4.02.91 5.36 2.37l2.8-2.8c-1.8-1.7-4.2-2.77-6.56-2.77c-5.4 0-9.8 4.4-9.8 9.8s4.4 9.8 9.8 9.8c5.4 0 9.8-4.4 9.8-9.8c0-.7-.08-1.38-.2-2.04z" />
          </svg>
          Войти с Google
        </button>

        <div className="mt-4 text-center text-gray-400 text-xs">
          Powered by AILBEE
        </div>
      </div>
    </div>
  );
};

const SummaryModal = ({ summary, onClose }) => {
  const { isDarkMode } = useTheme();
  const modalBg = isDarkMode ? 'bg-blue-900' : 'bg-white';
  const textColor = isDarkMode ? 'text-gray-100' : 'text-gray-900';
  const buttonBg = isDarkMode ? 'bg-blue-600' : 'bg-blue-500';
  const buttonHoverBg = isDarkMode ? 'hover:bg-blue-700' : 'hover:bg-blue-600';

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className={`relative p-8 rounded-lg shadow-lg w-full max-w-lg max-h-[90vh] overflow-y-auto ${modalBg}`}>
        <h3 className={`text-xl font-bold mb-4 ${textColor}`}>Суммирование книги</h3>
        <p className={`mb-6 ${textColor}`}>{summary}</p>
        <button
          onClick={onClose}
          className={`w-full py-2 px-4 rounded-md text-sm font-medium text-white ${buttonBg} ${buttonHoverBg} focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500`}
        >
          Закрыть
        </button>
      </div>
    </div>
  );
};


const Dashboard = () => {
  const { currentUser, auth, db, app } = useAuth();
  const { isDarkMode, toggleTheme } = useTheme();
  const [books, setBooks] = useState([]);
  const [loadingBooks, setLoadingBooks] = useState(false);
  const [booksError, setBooksError] = useState('');
  const [selectedSubject, setSelectedSubject] = useState(null);
  const [selectedEducationLevel, setSelectedEducationLevel] = useState(null);
  const [summarizedText, setSummarizedText] = useState(null);
  const [loadingSummary, setLoadingSummary] = useState(false);
  
  const subjects = ["Математика", "Биология", "История", "Физика", "Литература", "Химия", "Информатика", "Психология"];
  const educationLevels = ["Начальные классы", "Средняя школа", "Старшие классы", "Медицинский", "Архитектурный", "Литературный", "Математический", "Физический", "Химический", "Профессиональный"];

  const handleRequestNotificationToken = () => {
    // Проверяем, что и приложение Firebase (app), и объект аутентификации (auth) готовы
    if (app && auth) {
      // Передаем оба объекта в нашу функцию
      requestNotificationPermission(app, auth);
    } else {
      console.error("Firebase app or auth is not initialized yet.");
      alert("Ошибка: Firebase или система аутентификации не инициализированы. Невозможно запросить токен.");
    }
  };

  const fetchWithRetry = async (url, options, retries = 3, delay = 1000) => {
    for (let i = 0; i < retries; i++) {
      try {
        const response = await fetch(url, options);
        return response;
      } catch (error) {
        console.error(`Попытка ${i + 1} завершилась сетевой ошибкой:`, error);
        if (i < retries - 1) {
          await new Promise(res => setTimeout(res, delay * (2 ** i)));
        } else {
          throw error;
        }
      }
    }
    throw new Error('Все попытки повтора завершились неудачей.');
  };

  // Helper function for centralized API error handling
  const handleApiError = async (response, contextMessage, setErrorState) => {
    const errorPrefix = `Ошибка ${contextMessage}.`;
    console.error(`Dashboard: HTTP ошибка ${response.status}: ${response.statusText}`);

    try {
      const errorBody = await response.json(); // Attempt to parse JSON error
      const errorMessage = errorBody.detail || `Статус: ${response.status} ${response.statusText}`;
      console.error("Dashboard: Детали ошибки из ответа:", errorBody);
      setErrorState(`${errorPrefix} ${errorMessage}`);
    } catch (e) {
      const errorBodyText = await response.text(); // Fallback to text if JSON parsing fails
      setErrorState(`${errorPrefix} Статус: ${response.status} ${response.statusText}. Детали: ${errorBodyText}`);
    }
  };

  const fetchBooks = async (subject, educationLevel) => {
    if (!currentUser) {
      setLoadingBooks(false);
      return;
    }
    setLoadingBooks(true);
    setBooksError('');
    
    let fetchUrl = `${BACKEND_BASE_URL}/library/books`;
    const params = new URLSearchParams();
    if (subject) {
      params.append('subject', subject);
    }
    if (educationLevel) {
      const levelMap = {
        "Начальные классы": "primary",
        "Средняя школа": "middle",
        "Старшие классы": "high",
        "Медицинский": "university",
        "Архитектурный": "university",
        "Литературный": "university",
        "Математический": "university",
        "Физический": "university",
        "Химический": "university",
        "Профессиональный": "professional"
      };
      const mappedLevel = levelMap[educationLevel];
      if (mappedLevel) {
        params.append('education_level', mappedLevel);
      }
    }
    if (params.toString()) {
      fetchUrl += `?${params.toString()}`;
    }

    try {
      const idToken = await currentUser.getIdToken();
      console.log("Dashboard: Получение ID токена для запроса книг.");
      console.log("Dashboard: ПОПЫТКА ЗАПРОСА КНИГ по URL:", fetchUrl);
      console.log("Dashboard: ID Token (первые 20 символов):", idToken ? idToken.substring(0, 20) + "..." : "Нет токена");

      const response = await fetchWithRetry(fetchUrl, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${idToken}`,
        },
        mode: 'cors',
        credentials: 'include'
      });

      if (response.ok) {
        const result = await response.json();
        // Проверка ожидаемой структуры ответа
        if (!result || !Array.isArray(result.books)) {
          console.error("Dashboard: Получен некорректный формат ответа при загрузке книг:", result);
          setBooksError('Получен некорректный формат данных книг от сервера.');
          setBooks([]); // Очистить список книг при некорректном ответе
          return; // Прекратить выполнение при ошибке
        }
        setBooks(result.books);
        console.log("Dashboard: Книги успешно получены:", result.books);
      } else {
        // Использование централизованной функции обработки ошибок
        await handleApiError(response, "при получении списка книг", setBooksError);
      }
    } catch (err) {
      console.error("Dashboard: ОШИБКА FETCH - URL:", fetchUrl, "Ошибка:", err);
      setBooksError(err.message || 'Ошибка сети при получении книг.');
      console.error("Dashboard: Ошибка при получении книг:", err);
    } finally {
      setLoadingBooks(false);
    }
  };

  const handleSubjectClick = (subject) => {
    setSelectedSubject(subject);
    fetchBooks(subject, selectedEducationLevel);
  };

  const handleLevelClick = (level) => {
    setSelectedEducationLevel(level);
    fetchBooks(selectedSubject, level);
  };

  const handleSummarizeBook = async (bookDescription) => {
    if (!currentUser || !bookDescription) {
      setBooksError("Невозможно суммировать: нет пользователя или описания книги.");
      return;
    }
    setLoadingSummary(true);
    setSummarizedText(null);
    setBooksError('');

    try {
      const idToken = await currentUser.getIdToken();
      const fetchUrl = `${BACKEND_BASE_URL}/llm/summarize_text`;
      console.log("Dashboard: Запрос на суммирование LLM по URL:", fetchUrl);

      const response = await fetchWithRetry(fetchUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${idToken}`,
        },
        mode: 'cors',
        credentials: 'include',
        body: JSON.stringify({ text: bookDescription }),
      });

      if (response.ok) {
        const result = await response.json();
        // Проверка ожидаемой структуры ответа
        if (!result || typeof result.summary !== 'string') {
             console.error("Dashboard: Получен некорректный формат ответа при суммировании:", result);
             setBooksError('Получен некорректный формат суммирования от сервера.');
             setSummarizedText(null); // Очистить суммированный текст
             return; // Прекратить выполнение при ошибке
        }
        setSummarizedText(result.summary);
        console.log("Dashboard: Суммирование успешно:", result.summary);
      } else {
        await handleApiError(response, "при суммировании текста", setBooksError);
      }
    } catch (err) {
      console.error("Dashboard: ОШИБКА LLM SUMMARIZE FETCH - Ошибка:", err);
      setBooksError(err.message || 'Ошибка сети при суммировании текста.');
    } finally {
      setLoadingSummary(false);
    }
  };

  useEffect(() => {
    if (currentUser) {
      setSelectedSubject("Психология");
      fetchBooks("Психология", null);
    }
  }, [currentUser]);

  const handleSignOut = async () => {
    try {
      console.log("Dashboard: Попытка выхода.");
      await signOut(auth);
    } catch (err) {
      console.error("Dashboard: Ошибка выхода:", err);
      setBooksError("Ошибка выхода. Пожалуйста, попробуйте еще раз.");
    }
  };

  const handleOpenBook = (book) => {
    console.log("Dashboard: Открыть книгу:", book.title);
  };

  const bgColor = isDarkMode ? 'bg-blue-950' : 'bg-gray-50';
  const cardBg = isDarkMode ? 'bg-blue-900' : 'bg-white';
  const textColor = isDarkMode ? 'text-gray-100' : 'text-gray-900';
  const subTextColor = isDarkMode ? 'text-gray-300' : 'text-gray-700';
  const headingColor = isDarkMode ? 'text-gray-200' : 'text-gray-800';
  const borderColor = isDarkMode ? 'border-blue-700' : 'border-gray-200';
  const buttonBg = isDarkMode ? 'bg-blue-600' : 'bg-blue-500';
  const buttonHoverBg = isDarkMode ? 'hover:bg-blue-700' : 'hover:bg-blue-600';
  const bookCardBg = isDarkMode ? 'bg-blue-800' : 'bg-gray-100';
  const bookCardBorder = isDarkMode ? 'border-blue-700' : 'border-gray-200';
  const activeFilterBg = isDarkMode ? 'bg-blue-700' : 'bg-blue-400';
  const activeFilterText = 'text-white';

  return (
    <div className={`min-h-screen flex flex-col items-center p-4 ${bgColor}`}>
      <div className="w-full max-w-7xl flex justify-end mb-4">
        <button
          onClick={toggleTheme}
          className={`px-4 py-2 rounded-md text-sm font-medium ${isDarkMode ? 'bg-gray-700 text-white hover:bg-gray-600' : 'bg-gray-200 text-gray-800 hover:bg-gray-300'}`}
        >
          {isDarkMode ? 'Светлая тема' : 'Темная тема'}
        </button>
      </div>
      <div className={`w-full max-w-7xl p-8 rounded-lg shadow-lg flex flex-col md:flex-row ${cardBg}`}>
        <div className={`w-full md:w-1/5 p-4 border-r ${borderColor} md:h-[calc(100vh-100px)] overflow-y-auto`}>
          <h3 className={`text-xl font-semibold mb-4 ${headingColor}`}>Предметы</h3>
          {subjects.map(subject => (
            <button
              key={subject}
              onClick={() => handleSubjectClick(subject)}
              className={`w-full text-left p-3 mb-2 rounded-md transition-colors duration-200
                ${selectedSubject === subject ? `${activeFilterBg} ${activeFilterText}` : `${buttonBg} ${buttonHoverBg} text-white`}`
              }
            >
              {subject}
            </button>
          ))}
        </div>

        <div className="w-full md:w-3/5 p-4 md:h-[calc(100vh-100px)] overflow-y-auto">
          <h3 className={`text-xl font-semibold mb-4 text-center ${headingColor}`}>Библиотека учебников</h3>

          {booksError && (
            <p className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4" role="alert">
              {booksError}
            </p>
          )}

          {loadingBooks ? (
            <p className={`${subTextColor} text-center`}>Загрузка книг...</p>
          ) : books.length === 0 ? (
            <p className={`${subTextColor} text-center`}>
              Книги не найдены для выбранных фильтров. Пожалуйста, добавьте книги в библиотеку через бэкенд.
            </p>
          ) : (
            <div className="grid grid-cols-1 gap-4">
              {books.map(book => (
                <div key={book.bookId} className={`p-4 rounded-lg shadow-md ${bookCardBg} ${bookCardBorder}`}>
                  <h4 className={`text-lg font-semibold mb-2 ${textColor}`}>{book.title}</h4>
                  <p className={`text-sm ${subTextColor}`}>Автор: {book.author}</p>
                  <p className={`text-sm ${subTextColor}`}>Предмет: {book.subject}</p>
                  <p className={`text-sm ${subTextColor}`}>Уровень: {book.educationLevel}</p>
                  <p className={`text-sm ${subTextColor} mb-4`}>Описание: {book.description ? book.description.substring(0, 100) + '...' : 'Нет описания'}</p>
                  <button
                    onClick={() => handleOpenBook(book)}
                    className={`mt-2 w-full py-2 px-4 rounded-md text-sm font-medium text-white ${buttonBg} ${buttonHoverBg} focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500`}
                  >
                    Открыть книгу
                  </button>
                  {book.description && (
                    <button
                      onClick={() => handleSummarizeBook(book.description)}
                      className={`mt-2 w-full py-2 px-4 rounded-md text-sm font-medium text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500`}
                      disabled={loadingSummary}
                    >
                      {loadingSummary ? 'Суммирование...' : '✨ Суммировать книгу'}
                    </button>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        <div className={`w-full md:w-1/5 p-4 border-l ${borderColor} md:h-[calc(100vh-100px)] overflow-y-auto`}>
          <h3 className={`text-xl font-semibold mb-4 ${headingColor}`}>Уровни</h3>
          {educationLevels.map(level => (
            <button
              key={level}
              onClick={() => handleLevelClick(level)}
              className={`w-full text-left p-3 mb-2 rounded-md transition-colors duration-200
                ${selectedEducationLevel === level ? `${activeFilterBg} ${activeFilterText}` : `${buttonBg} ${buttonHoverBg} text-white`}`
              }
            >
              {level}
            </button>
          ))}
        </div>
      </div>

      {/* === ИЗМЕНЕНИЯ ЗДЕСЬ === */}
      <div className="mt-8 text-center">
        <button
          onClick={handleSignOut}
          className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
        >
          Выйти
        </button>

        {/* ВОТ ЭТА КНОПКА */}
        <button
          onClick={handleRequestNotificationToken}
          className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 ml-4"
        >
          Получить токен для уведомлений
        </button>
      </div>

      {summarizedText && <SummaryModal summary={summarizedText} onClose={() => setSummarizedText(null)} />}
    </div>
  );
};

const App = () => {
  const { currentUser } = useAuth();

  return (
    <div className="App">
      {currentUser ? <Dashboard /> : <Auth />}
    </div>
  );
}

export default function WrappedApp() {
  return (
    <AuthProvider>
      <ThemeProvider>
        <App />
      </ThemeProvider>
    </AuthProvider>
  );
}
