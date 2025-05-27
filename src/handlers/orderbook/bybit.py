from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import chex

def init() -> Tuple[chex.Array, chex.Array]:
    """
    Initializes empty ask and bid arrays.
    """
    asks = jnp.empty((0, 2), dtype=jnp.float32)
    bids = jnp.empty((0, 2), dtype=jnp.float32)
    return asks, bids

def sort_book(asks: chex.Array, bids: chex.Array, levels: int) -> Tuple[chex.Array, chex.Array]:
    """
    Sorts the ask orders in ascending order and bid orders in descending order by price.
    Only keeps the top 500 orders in each.
    """
    asks = asks[asks[:, 0].argsort()][:levels]
    bids = bids[bids[:, 0].argsort()[::-1]][:levels]
    return asks, bids

def update_book(asks_or_bids: chex.Array, data: chex.Array) -> chex.Array:
    """
    Updates the specified order book (asks or bids) with new data.

    Parameters
    ----------
    asks_or_bids : chex.Array
        The current state of either the asks or bids in the order book.
    data : chex.Array
        New data to be integrated into the asks or bids. Each element is a [price, quantity] pair.

    Returns
    -------
    chex.Array
        The updated asks or bids array.
    """
    for price, qty in data:
        # Remove orders with the specified price
        asks_or_bids = asks_or_bids[asks_or_bids[:, 0] != price]
        # Add new or updated order if quantity is greater than zero
        if qty > 0:
            asks_or_bids = jnp.vstack((asks_or_bids, jnp.array([price, qty])))

    return asks_or_bids

def process_snapshot(asks: chex.Array, bids: chex.Array, levels: int) -> Tuple[chex.Array, chex.Array]:
    """
    Processes and initializes the order book with a snapshot of asks and bids.

    Parameters
    ----------
    asks : chex.Array
        A list of ask orders, each represented as [price, quantity].
    bids : chex.Array
        A list of bid orders, each represented as [price, quantity].
    """
    asks = jnp.array(asks, dtype=jnp.float32)
    bids = jnp.array(bids, dtype=jnp.float32)
    asks, bids = sort_book(asks, bids, levels)
    return asks, bids

def process(recv: Dict, current_asks: chex.Array, current_bids: chex.Array, levels: int) -> Tuple[chex.Array, chex.Array]:
    """
    Handles incoming WebSocket messages to update the order book.

    Parameters
    ----------
    recv : Dict
        The incoming message containing either a snapshot or delta update of the order book.
    current_asks : chex.Array
        The current state of the ask orders in the order book.
    current_bids : chex.Array
        The current state of the bid orders in the order book.
    levels : int
        The number of levels to keep in the order book.
    """
    asks = jnp.array(recv["data"]["a"], dtype=jnp.float32)
    bids = jnp.array(recv["data"]["b"], dtype=jnp.float32)

    if recv["type"] == "snapshot":
        asks, bids = process_snapshot(asks, bids, levels)
        
    elif recv["type"] == "delta":
        asks = update_book(current_asks, asks)
        bids = update_book(current_bids, bids)
        asks, bids = sort_book(asks, bids, levels)

    return asks, bids

def prepare_for_storing(asks: chex.Array, bids: chex.Array) -> chex.Array:
    """
    Prepares the order book for storing in a CSV file.
    """
    return jnp.hstack((asks, bids)).flatten()

def write_book_to_csv(obs: chex.Array, filename: str) -> None:
    """
    Writes the batch of order books to a CSV file in the format:
    level 1 ask price, level 1 ask qty, level 1 bid price, level 1 bid qty, 
    level 2 ask price, level 2 ask qty, level 2 bid price, level 2 bid qty, ...
    
    The function appends to the file if it exists, or creates it if it doesn't.
    No headers are included in the file.
    Args:
        obs, chex.Array: The array to be written, of shape [batch, 4xnum_levels]
        filename, str: The path to the file to be written
    """
    obs: list[list[float]] = obs.tolist()
    ob = [','.join([str(x) for x in ob]) + '\n' for ob in obs]
    with open(filename, "a", encoding="utf-8") as f:
        # Write data
        f.writelines(ob)
