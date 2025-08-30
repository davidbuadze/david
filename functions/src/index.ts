// david/functions/src/index.ts
import * as functions from "firebase-functions";
import * as admin from "firebase-admin";

// Инициализация Firebase Admin SDK, если еще не инициализировано
if (!admin.apps.length) {
  admin.initializeApp();
}

const db = admin.firestore();

// Ваша существующая функция helloWorld
export const helloWorld = functions.https.onRequest((request, response) => {
  functions.logger.info("Hello logs!", {structuredData: true});
  response.send("Hello from Firebase!");
});

// НОВАЯ Cloud Function для обработки экспортированных данных
export const processExportedUserData = functions.https.onRequest(async (request, response) => {
  // Проверка метода HTTP запроса (ожидаем POST от расширения)
  if (request.method !== 'POST') {
    response.status(405).send('Method Not Allowed');
    return;
  }

  try {
    // Получение данных из тела запроса (ожидаем JSON)
    const exportData = request.body;

    // --- логика обработки экспортированных данных здесь ---
    // Примеры действий: логирование, учет в Firestore, анализ размера файлов

    console.log(`Получены данные о завершении экспорта: ${exportData.exportId}`);
    console.log(`Статус экспорта: ${exportData.status}`);
    console.log(`Количество экспортированных файлов: ${exportData.exportedFiles ? exportData.exportedFiles.length : 0}`);

    if (exportData.status === 'SUCCESS') {
      // --- Попытка извлечь UID пользователя из данных экспорта ---
      let userId = null;
      if (exportData.exportParams?.firestorePath === 'users/{UID}' && exportData.exportedFiles && exportData.exportedFiles.length > 0) {
         // Если экспортировался путь users/{UID}, и есть файлы,
         // попробуем извлечь UID из имени файла (пример: user-UID.json)
         const firstFilePath = exportData.exportedFiles[0].filePath;
         const userIdMatch = firstFilePath.match(/users-([^/]+)\.json/); // Извлекаем то, что между 'users-' и '.json'
         if (userIdMatch && userIdMatch[1]) {
           userId = userIdMatch[1];
           console.log(`Извлечен UID пользователя из имени файла: ${userId}`);
         } else {
            console.warn(`Не удалось извлечь UID пользователя из имени файла: ${firstFilePath}`);
         }
      } else {
          console.log("Экспорт, вероятно, не связан с конкретным пользователем или формат пути Firestore отличается.");
          // Здесь можно добавить логику для других сценариев определения пользователя
      }
      // --- Конец попытки извлечения UID ---
      // --- *** Важное примечание по извлечению UID:** Паттерн `users-([^/]+)\.json` 
      // для извлечения UID из имени файла является **предположением** о том,
      // как расширение "Export User Data" называет экспортированные файлы при экспорте пути 
      // `users/{UID}`. **Вам нужно будет протестировать это на практике и, возможно, 
      // скорректировать регулярное выражение**, чтобы оно точно соответствовало именам файлов,
      // создаваемых расширением.

      const exportRecord = {
        exportId: exportData.exportId,
        status: exportData.status,
        timestamp: new Date(exportData.timestamp),
        firestorePath: exportData.exportParams?.firestorePath || null,
        cloudStorageBucket: exportData.exportParams?.cloudStorageBucket || null,
        cloudStorageFolder: exportData.exportParams?.cloudStorageFolder || null,
        exportedFiles: exportData.exportedFiles || [],
        processedAt: admin.firestore.FieldValue.serverTimestamp()
      };
      await db.collection('export_logs').doc(exportData.exportId).set(exportRecord);
      console.log(`Информация об экспорте ${exportData.exportId} записана в Firestore.`);

      let totalSizeBytes = 0;
      if (exportData.exportedFiles) {
        totalSizeBytes = exportData.exportedFiles.reduce((sum, file) => sum + (file.sizeBytes || 0), 0);
      }
      console.log(`Общий размер экспортированных данных: ${totalSizeBytes} байт.`);
      // Здесь можно добавить логику учета расхода ресурсов
    } else {
      console.error(`Экспорт ${exportData.exportId} завершился с ошибкой. Детали: ${JSON.stringify(exportData)}`);
    }

    // --- Конец логики ---

    response.status(200).send('Export data processed successfully');

  } catch (error) {
    console.error('Ошибка при обработке данных экспорта:', error);
    response.status(500).send('Error processing export data');
  }
});
