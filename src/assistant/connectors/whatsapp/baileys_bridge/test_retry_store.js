/**
 * Test script for Baileys Bridge BoundedMessageStore, getMessage, and retry cache.
 */

const assert = require("assert");
const { BoundedMessageStore, messageStore, msgRetryCounterCache } = require("./index.js");

async function runTests() {
  console.log("Running Baileys Bridge message store and retry tests...");

  // Test 1: Outbound message is stored
  const testStore = new BoundedMessageStore(10);
  const outMsgId = "OUT_MSG_12345";
  const outPayload = {
    conversation: "Hello from assistant",
  };
  testStore.set(outMsgId, outPayload);
  assert.strictEqual(testStore.has(outMsgId), true, "Outbound message should be in store");

  // Test 2: getMessage retrieves stored outbound message
  const getMessageMock = async (key) => {
    const item = testStore.get(key?.id);
    return item?.message || item;
  };

  const retrieved = await getMessageMock({ id: outMsgId, remoteJid: "919172767219@s.whatsapp.net" });
  assert.deepStrictEqual(retrieved, outPayload, "getMessage should retrieve outbound payload");

  // Test 3: Unknown message ID returns undefined
  const notFound = await getMessageMock({ id: "UNKNOWN_MSG_999", remoteJid: "919172767219@s.whatsapp.net" });
  assert.strictEqual(notFound, undefined, "Unknown message ID must return undefined");

  // Test 4: Inbound message is stored
  const inMsgId = "IN_MSG_67890";
  const inPayload = {
    extendedTextMessage: { text: "User prompt" },
  };
  testStore.set(inMsgId, inPayload);
  assert.strictEqual(testStore.has(inMsgId), true, "Inbound message should be in store");
  const retrievedInbound = await getMessageMock({ id: inMsgId });
  assert.deepStrictEqual(retrievedInbound, inPayload, "getMessage should retrieve inbound payload");

  // Test 5: Store remains bounded
  const smallStore = new BoundedMessageStore(3);
  smallStore.set("id_1", { conversation: "1" });
  smallStore.set("id_2", { conversation: "2" });
  smallStore.set("id_3", { conversation: "3" });
  assert.strictEqual(smallStore.size, 3);

  // Adding 4th should evict id_1
  smallStore.set("id_4", { conversation: "4" });
  assert.strictEqual(smallStore.size, 3, "Size must remain bounded at 3");
  assert.strictEqual(smallStore.has("id_1"), false, "Oldest message id_1 should be evicted");
  assert.strictEqual(smallStore.has("id_2"), true);
  assert.strictEqual(smallStore.has("id_3"), true);
  assert.strictEqual(smallStore.has("id_4"), true);

  // Test 6: Retry counter cache exists and functions
  assert.ok(msgRetryCounterCache, "msgRetryCounterCache must exist");
  msgRetryCounterCache.set("msg1:user1", 1);
  assert.strictEqual(msgRetryCounterCache.get("msg1:user1"), 1, "msgRetryCounterCache get/set must work");
  msgRetryCounterCache.del("msg1:user1");
  assert.strictEqual(msgRetryCounterCache.get("msg1:user1"), undefined);

  // Test 7: Recreated socket simulation still accesses the same in-process store
  messageStore.clear();
  messageStore.set("SHARED_MSG_1", { conversation: "persistent in process" });

  // Simulate socket 1
  const socket1GetMessage = async (key) => {
    const item = messageStore.get(key?.id);
    return item?.message || item;
  };
  assert.ok(await socket1GetMessage({ id: "SHARED_MSG_1" }));

  // Simulate socket 2 created after reconnect
  const socket2GetMessage = async (key) => {
    const item = messageStore.get(key?.id);
    return item?.message || item;
  };
  const res2 = await socket2GetMessage({ id: "SHARED_MSG_1" });
  assert.deepStrictEqual(res2, { conversation: "persistent in process" }, "Recreated socket must access shared store");

  console.log("All Baileys Bridge message store and retry tests passed successfully!");
}

runTests().catch((err) => {
  console.error("Test failed:", err);
  process.exit(1);
});
