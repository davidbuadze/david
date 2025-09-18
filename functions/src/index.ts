// ნიშანი
import * as functions from "firebase-functions/v1";
import * as admin from "firebase-admin";
import { getMessaging } from "firebase-admin/messaging";
import { Message } from "firebase-admin/messaging";

admin.initializeApp();

const db = admin.firestore();
const messaging = getMessaging();

// V1 Auth Triggers
export const createUserData = functions.auth.user().onCreate((user) => {
  const newUser = {
    email: user.email,
    displayName: user.displayName,
    photoURL: user.photoURL,
    uid: user.uid,
    createdAt: user.metadata.creationTime,
    role: "user",
  };
  return db.collection("users").doc(user.uid).set(newUser);
});

export const cleanupUserData = functions.auth.user().onDelete((user) => {
  return db.collection("users").doc(user.uid).delete();
});

// V1 Realtime Database Triggers with explicit instance
export const onNewUserStatus = functions.database.instance("ailbee").ref("/statuses/{statusId}").onCreate(async (snapshot, context) => {
    const status = snapshot.val();
    const userRef = db.collection("users").doc(status.uid);
    const userSnap = await userRef.get();
    const userData = userSnap.data();

    if (!userData || !userData.fcmTokens) {
        console.log("User data or FCM tokens not found for UID:", status.uid);
        return;
    }

    const tokens = Object.keys(userData.fcmTokens);
    if (tokens.length === 0) {
        console.log("User has no FCM tokens, skipping notification.");
        return;
    }

    const messages: Message[] = tokens.map(token => ({
        token: token,
        notification: {
          title: `New status from ${userData.displayName}`,
          body: status.text,
          icon: userData.photoURL,
        }
    }));

    const response = await messaging.sendEach(messages);
    const tokensToRemove: Promise<any>[] = [];
    response.responses.forEach((result, index) => {
        const error = result.error;
        if (error) {
            console.error("Failure sending notification to", tokens[index], error);
            if (error.code === "messaging/invalid-registration-token" ||
                error.code === "messaging/registration-token-not-registered") {
                tokensToRemove.push(userRef.update({
                    [`fcmTokens.${tokens[index]}`]: admin.firestore.FieldValue.delete()
                }));
            }
        }
    });

    return Promise.all(tokensToRemove);
 });

export const onUserStatusChange = functions.database.instance("ailbee").ref("/statuses/{statusId}").onUpdate(async (change, context) => {
    const after = change.after.val();
    const before = change.before.val();

    if(after.likes === before.likes) {
        console.log("Likes haven't changed, skipping notification.");
        return;
    }

    const userRef = db.collection("users").doc(after.uid);
    const userSnap = await userRef.get();
    const userData = userSnap.data();

    if (!userData || !userData.fcmTokens) {
        console.log("User data or FCM tokens not found for UID:", after.uid);
        return;
    }

    const tokens = Object.keys(userData.fcmTokens);
    if (tokens.length === 0) {
        console.log("User has no FCM tokens, skipping notification.");
        return;
    }

    const messages: Message[] = tokens.map(token => ({
        token: token,
        notification: {
          title: `Status update from ${userData.displayName}`,
          body: `Your status now has ${after.likes} likes!`,
          icon: userData.photoURL,
        }
    }));

    const response = await messaging.sendEach(messages);
    const tokensToRemove: Promise<any>[] = [];
    response.responses.forEach((result, index) => {
        const error = result.error;
        if (error) {
            console.error("Failure sending notification to", tokens[index], error);
            if (error.code === "messaging/invalid-registration-token" ||
                error.code === "messaging/registration-token-not-registered") {
                tokensToRemove.push(userRef.update({
                    [`fcmTokens.${tokens[index]}`]: admin.firestore.FieldValue.delete()
                }));
            }
        }
    });

    return Promise.all(tokensToRemove);
});
