# The Wallet tap reminder (iPhone)

When you tap a card in Apple Wallet, a Shortcut asks CardPilot what each of your cards earns at that
kind of shop and shows it as a notification. It states earn rates only. It never tells you which card
to use, and nothing about the purchase is stored.

The endpoint:

```
GET /api/tap?country=us&merchant=Starbucks&cards=Amex Gold,Citi Double Cash
```

returns one line of plain text, for example:

```
Starbucks looks like dining. Earn on dining: American Express Gold Card 4%, Citi Double Cash Card 2%.
Caps may apply. Rates as of 2026-09-29. Education only.
```

`cards` is optional. Without it you get the three highest rates in the catalog for that category.
`country` is `us` or `in`. Card names can be short, like `Amex Gold` or `HDFC Millennia`.

## 1. Make CardPilot reachable from the phone

The phone cannot reach `localhost` on your Mac. Pick one.

- **Same Wi-Fi, for trying it.** On the Mac run
  `uv run uvicorn cardpilot.api:app --host 0.0.0.0 --port 8000`, find the Mac's address in
  System Settings, Wi-Fi, Details (like `192.168.1.20`), and use `http://192.168.1.20:8000`.
- **Always on.** Deploy the Docker image to any host that runs containers (Render, Fly.io, Railway,
  Google Cloud Run) and use its `https://` address.

Check it from Safari on the phone: `http://YOUR-ADDRESS/api/tap?country=us&merchant=starbucks`.

## 2. Build the automation

1. Open **Shortcuts**, then **Automation**, then **+** (New Automation).
2. Choose **Transaction** (Wallet). Pick the cards it should run for and leave the categories on
   "any". Set it to **Run Immediately** and turn off **Notify When Run**.
3. Tap **New Blank Automation** and add these actions:
   1. **Text** with your card names, like `Amex Gold,Citi Double Cash`.
   2. **URL** with
      `http://YOUR-ADDRESS/api/tap?country=us&cards=[Text]&merchant=[Shortcut Input, Merchant]`.
      Insert the two variables with the variable picker (tap the Shortcut Input variable and choose
      its Merchant detail).
   3. **Get Contents of URL** (method GET).
   4. **Show Notification** with **Contents of URL** as the body.
4. Tap a card at a shop, or add a test transaction, and the notification appears.

## Limits

- The merchant name from Wallet can be cryptic (`SQ *BLUE BOTTLE 0231`). Unknown names fall back to
  everyday rates, and the notification says so.
- Categories come from simple name matching in `src/cardpilot/api.py` (`MERCHANT_WORDS`). Add the
  shops you use there.
- The endpoint is rate limited per address, like the rest of the API.
